# 第三堂課 講稿／索引 — 推論引擎單機篇：八個問題，兩種解法（SGLang × vLLM）

投影片：[../slides/class3-engine-single-node.html](../slides/class3-engine-single-node.html)（27 頁）｜重建：`cd ../slides/build && node generate_class3.js`
互動地圖：[serving-map.html](../interactive/serving-map.html)（第 16 頁指引，**本堂只用模式 1–4**，模式 5 PD 分離留給第四堂）

> **主幹＝問題導向，而且一個問題兩種寫法。**
> 那八個問題**不是 SGLang 專有的，是任何推論引擎都會撞到的**——SGLang 的發展史剛好把它們依序列了出來。所以最好的學法是：**一個問題、兩種解法，對比之後就看得到兩家哲學的差異。**
> 本堂做 ①–④（單機），⑤–⑧ 在[第四堂](class4-models-to-racks.md)後半的請求路徑裡逐站出現（多機）。前置：第一堂 roofline 與記憶體階層、第二堂多卡平行與互連。
>
> **接上一堂的口白**：「上一堂我們把卡連起來了，但留下一個沒解的問題：decode 要 batch 拉到幾百才吃得滿算力，而 DP 救不了單一請求的延遲。今天先退回**一台機器**——卡連起來之前，一台都還沒餵飽。」（第二堂的平行偏訓練視角；推論的多機版本在第四堂。）

---

## 八個問題全景（第 2 頁）

| # | 遇到的問題 | SGLang | vLLM | 場次 |
|---|---|---|---|---|
| ① | LLM 程式（多次呼叫、分支、工具）難寫又跑不快 | 前端 DSL | **—**（沒有對應物） | 本堂 |
| ② | 這些程式天然共享大量前綴，卻被反覆重算 | radix tree + cache-aware 排程 | 鏈式雜湊表 APC | 本堂 |
| ③ | 結構化輸出逐 token 檢查太慢、格式仍不保證 | XGrammar + jump-forward | XGrammar（非同步編譯） | 本堂 |
| ④ | GPU 一步只要 5–10 ms，CPU 排程成了瓶頸 | zero-overhead scheduler | 多進程 + async 排程 | 本堂 |
| ⑤ | 大 MoE 單機放不下、專家負載不均 | 大規模 EP + DeepEP/EPLB | 支援 EP | 第四堂 |
| ⑥ | prefill 與 decode 互相干擾 | 內建 PD 分離 | KV connector 抽象 | 第四堂 |
| ⑦ | 多副本：局部性 vs 負載均衡 | 自帶 cache-aware router | 生態系（Dynamo / llm-d） | 第四堂 |
| ⑧ | 副本掛掉，請求與 KV 怎麼辦 | — | — | 第四堂 |

> **講法**：先把八個問題全攤出來，讓聽眾看到「這是一條線，不是一堆功能」，而且**兩家都在解同一批問題**。然後特別指出 ① 那一列——**vLLM 沒有對應物，那不是缺漏，是職責邊界的選擇**（第 8 頁會展開）。

---

## 頁面地圖（27 頁）

| 頁 | 內容 |
|---|---|
| 1 | 標題：八個問題，兩種解法 |
| 2 | **全景**：八個問題 × 兩家的寫法（線以上本堂） |
| 3 | **先給結論**：兩家的哲學差異（後面逐題驗證） |
| **地基** | |
| 4 | 地基①：batch=1 的 decode 只用 0.34% 算力 |
| 5 | 地基②：AI ≈ B；四個問題各擋住 B 的哪一段 |
| **問題① 程式難平行** | |
| 6 | 問題①：Python 迴圈把依賴關係寫死在控制流裡 |
| 7 | SGLang 的解法：前端 DSL 把程式描述成執行圖 |
| 8 | **對比①：vLLM 沒有對應物——而那是刻意的** |
| 9 | 橋：好寫的並行程式會製造大量重複計算 |
| **問題② 前綴重算** | |
| 10 | 問題②：要複用前綴，得先解決 KV 怎麼放（128 KB/token、預留浪費） |
| 11 | **共同地基**：分頁——兩家都這樣放 KV |
| 12 | **對比②a 索引層**：radix tree vs 鏈式雜湊表 |
| 13 | **對比②b 排程**：cache-aware 主動 vs 被動命中 |
| 14 | **共同地基**：continuous batching |
| 15 | 所以問題②該選哪一家？（含三個容易被忽略的選型因素） |
| 16 | 🧭 互動環節：serving-map 模式 1–4（6.8% → 81%） |
| **問題③ 輸出不可控** | |
| 17 | 問題③：只靠 prompt 說「請輸出 JSON」不可靠（四種翻車） |
| 18 | **共同地基**：FSM + 位元遮罩（兩家預設都用 XGrammar） |
| 19 | **對比③**：jump-forward decoding（語法允許時一次吐多個 token） |
| 20 | 問題② 與 ③ 的關係：正交，但有一條紅線 |
| **問題④ CPU 成瓶頸** | |
| 21 | 問題④：GPU 快到讓 CPU 變成瓶頸 + 共同地基 CUDA Graph |
| 22 | **對比④**：zero-overhead scheduler vs 多進程 + async |
| 23 | 雙胞胎：TTFT vs ITL → chunked prefill（兩家都有） |
| **天花板** | |
| 24 | 天花板①：投機解碼 / MTP（兩家都有，draft 來源不同） |
| 25 | 天花板②：量化 |
| 26 | **彙整：四個問題 × 兩種寫法**（本堂核心產出） |
| 27 | 帶走三句話 |

---

## 兩家的哲學差異（第 3 頁，全堂的鉤子）

| | SGLang | vLLM |
|---|---|---|
| 自我定位 | **「結構化 LLM 程式」的執行引擎** | **「模型服務」的基礎設施** |
| 第一性問題 | 一個會分支、會重試、會反覆用同樣前綴與語法的 LLM 程式，怎麼跑得最快？ | 任何模型、任何硬體、任何部署形態，怎麼都能穩定地服務起來？ |
| 手段 | DSL 描述執行圖、radix tree 索引、cache-aware 排程主動提高命中率、jump-forward 跳過確定的 token | 模型／硬體後端覆蓋最廣、KV connector 與語法後端皆可插拔、路由與編排交給生態系、V1 重寫壓低 CPU 開銷 |

> **一句話**：SGLang 在「前綴與語法重複」這條路上**挖得深**；vLLM 在「什麼都能跑」這個面上**鋪得廣**。
> 不是誰比較好，是**最佳化的目標函數不同**。這頁先講，後面每一題都會驗證一次。

**更精準的說法（第 27 頁會回收）**：兩家的差異幾乎都在**「要不要替使用者猜工作負載」**。SGLang 猜你會重複用前綴與語法，猜對了就贏很多；vLLM 不猜。

---

## 地基（第 4–5 頁）

### 推導 1 — batch=1 的 decode 只用 0.34% 算力

權重 fp16、參數量 $N$：搬 $2N$ bytes、算 $2N$ FLOPs → **AI = 1 FLOP/Byte**。
H100 ridge point = 990 TFLOPS ÷ 3.35 TB/s ≈ **296** → 利用率上限 ≈ **0.34%**。（A100 ≈ 0.6%，**卡越強越浪費**。）這就是第一堂開場謎題「<5%」算到底的值——本頁是回顧，一分鐘帶過即可，推導細節指回第一堂第 10–13、18 頁。

### 推導 2 — AI ≈ B

batch = B 時權重仍只讀一遍，FLOPs 變 B 倍 → **AI ≈ B**。既然 ridge ≈ 296，**B 要拉到幾百**。
Llama-3-8B on H100：batch=1 ≈ 208 tok/s → batch=64 ≈ **13,000 tok/s**（步時幾乎不變）。

### 四個問題各自擋住 B 的哪一段

| 問題 | 它怎麼擋住 B |
|---|---|
| ① 程式難平行 | 根本沒有那麼多請求同時進來——runtime 看不見可平行的分支 |
| ② 前綴重算 | HBM 被浪費的 KV 佔住 → 放不下更多條；算力浪費在不必算的東西上 |
| ③ 輸出不可控 | 每步的語法檢查是 CPU 上的序列工作 → 拖慢整批 |
| ④ CPU 成瓶頸 | GPU 算完在等 CPU 決定下一步 → 有效 batch 再大也填不滿時間軸 |

---

## 問題① 程式難平行（第 6–9 頁）

### 痛點
多次呼叫、分支、工具使用、self-consistency 取樣裡很多步驟其實可以同時跑，但用 `for` 迴圈寫出來之後 **runtime 只看得到「一個接一個的請求」**。加上 GIL，前端本身也不擅長真正的並行。

三個痛：①寫的人痛（膠水程式又臭又長）②跑的人痛（無從批處理）③**更痛的是這些請求高度共享前綴，但沒人告訴 runtime**。

### SGLang 的解法：前端 DSL
`sgl.gen` / `sgl.fork` / `sgl.select` **不是語法糖**——它讓 runtime 看見「什麼依賴什麼」。

```
for 迴圈    → runtime 看到：4 個先後到達的請求
sgl.fork()  → runtime 看到：1 個共享前綴 + 3 條可平行分支
```

名字就是這裡來的：**S**tructured **G**eneration **Lang**uage——它先是一個語言，才是一個引擎。

### 對比①：vLLM 沒有對應物（第 8 頁，本堂第一個對比）

| | SGLang：有一層語言 | vLLM：只有 API |
|---|---|---|
| 介面 | DSL 描述執行圖 | Python `LLM` 類 + OpenAI 相容 server |
| 平行化 | runtime 看得見分支結構，主動批處理 | 交給呼叫端（asyncio、批次提交）或上層框架（LangChain、DSPy…） |
| 前綴 | fork 的共享前綴直接餵給 radix tree | 靠 APC 被動命中 |
| 取捨 | 要學一套 DSL、綁定這個 runtime | 任何客戶端都能用，零學習成本 |

**為什麼這不是「缺漏」？** 因為兩家對「引擎的職責邊界」判斷不同：
- vLLM 認為「怎麼組織 LLM 程式」是**應用層**的事，引擎只該把單一請求服務好
- SGLang 認為**程式結構是效能資訊**，不交給引擎就浪費了

> **誠實補充**：實務上多數人把 SGLang 當純 server 用、從沒碰過 DSL——所以這一層在日常使用上常常是**沒被啟用的器官**。但它解釋了 SGLang 為什麼那麼在意前綴共享。

### 橋（第 9 頁）
一個 fork 出 8 條分支、system prompt 2,000 token 的程式 → 同一段 prefill 被算了 8 遍。
**而且真實流量本來就長這樣，不用 DSL 也一樣**：RAG、多輪對話、few-shot、agent。
→ 所以問題②是**兩家都必須解**的，只是 SGLang 因為 DSL 被逼得更早、更徹底。

---

## 問題② 前綴重算（第 10–16 頁，本堂最厚的一段）

### KV cache 有多大（第 10 頁）

```
每 token = 2 (K,V) × n_kv_heads × head_dim × n_layers × dtype_bytes
```
- Llama-3-8B（GQA 8 KV heads、128 dim、32 層、fp16）＝ **128 KB/token** → 8K 序列 = **1 GB**
- MHA 32 heads 就是 512 KB/token（**GQA 的價值就在這 4×**）

問題出在**不知道會生成多長**：照 `max_seq_len` 連續預留 → 多數請求只用 20–30%，**60–80% 是死重** → 放得下的條數被砍到 1/4。

### 共同地基：分頁（第 11 頁）

固定大小 block（16 token）、不連續、用完才要、block table 做映射；浪費 60–80% → **<4%**。
**vLLM 的 PagedAttention 讓它出名，但 SGLang 底下同樣是分頁式記憶體池——這一層兩家沒有分歧。**
關鍵副產品：**block 可跨請求共享** → 打開複用的門。

> ⚠️ **順帶破除一個迷思**：網路文章常把「RadixAttention vs PagedAttention」寫成競品，那是**把兩層混為一談**——分頁是「KV 怎麼放」（記憶體配置層，兩家一致），radix tree 是「放好的 block 怎麼被找到」（索引層，兩家不同）。
> **舊版投影片曾為此獨立一頁澄清；改成兩家對照之後，版面本身就已經把這件事講完了（共同地基 → 對比索引層），所以那頁刪掉，只留這一句。**

### 對比②a 索引層：radix tree vs 鏈式雜湊表（第 12 頁）

同一個問題：「這個新請求的前綴，之前算過嗎？算到哪？」

| | SGLang：radix tree | vLLM：鏈式雜湊（APC） |
|---|---|---|
| 結構 | 前綴樹，跨所有快取序列共用一棵 | 平坦雜湊表，每個 block 各自可尋址 |
| 鍵 | token 序列 | `hash(前一塊 hash + 本塊 tokens)` |
| 查詢 | 一次 **O(D)** 最長前綴匹配 | 逐 block 查表 **O(1)** |
| 粒度 | **分叉點可落在任意 token**，不必對齊 block 邊界 | **命中必須對齊 16-token 邊界** |
| 其他 | LRU 淘汰；可分層下放 CPU／磁碟 | V1 常數時間淘汰、預設開啟、命中率 0% 時幾乎零開銷 |

> **差異的本質**：**樹能表達「任意長度的共同開頭」，雜湊表只能表達「對齊的區塊」。** 所以極端共享場景 radix 佔優，一般場景兩者接近。

⚠️ **前綴匹配的常見誤解（一定要講）**：匹配是**嚴格按 token 順序從頭比對**。「怎麼退貨」和「運費怎麼算」都含「怎麼」，但**不會被合併**——只有共同**開頭**才算數。理由：KV 是按位置逐 token 累積的，第 3 個 token 的 K/V 依賴前面兩個。**位置一樣、前文一樣，K/V 才一樣。**

### 對比②b 排程：主動 vs 被動（第 13 頁，兩家差最多的一點）

有了索引還不夠——**請求「以什麼順序進來」也會改變命中率**：

```
亂序：A(前綴X) → B(前綴Y) → C(前綴X)   ← X 的快取可能已被 LRU 淘汰
排過：A(前綴X) → C(前綴X) 命中！ → B(前綴Y)
```

| | SGLang：cache-aware 排程 | vLLM：被動命中 |
|---|---|---|
| 做法 | 主動把同前綴的請求排在一起送；v0.4 起還有 cache-aware load balancer | FCFS / priority 為主，不按內容重排 |
| 命中率 | 引擎**主動經營快取** | 取決於流量自然的到達順序 |
| 哲學 | 替你猜工作負載 | 不猜，交給上層 |

> **實測**（前綴共用重、c=50）：SGLang 的 TTFT **p50 低約 37%、p95 低約 41%**——**差距主要來自這一頁，不只是資料結構。**

### 共同地基：continuous batching（第 14 頁）
排程單位從「一個請求」改成「一次 forward」。兩家都有；vLLM V1 更把 prefill/decode 統一成一個 `{req_id: num_tokens}` 排程字典。

> **三隻腳**：分頁撐大「B 的上限」、continuous batching 撐滿「實際值」、前綴複用省掉「不必算的」。

### 怎麼選（第 15 頁）

| 前綴共用 > 60% → SGLang | 前綴各自獨立 → 看生態成熟度 |
|---|---|
| RAG、多輪對話、few-shot、agent、`sgl.fork()` 分支 | 客服（不相干問題）、批次翻譯、一次性摘要 → 兩家吞吐差 <5%，選支援更廣的 |

**三個容易被忽略的選型因素**：
1. **硬體** —— 非 NVIDIA（AMD / TPU / Gaudi / CPU）→ vLLM 覆蓋明顯更廣
2. **模型** —— 剛出的新架構誰先支援？通常 vLLM 廣、SGLang 對 DeepSeek 系特別快
3. **多租戶安全** —— 前綴快取跨使用者共享會讓 TTFT 洩漏「這段 prompt 是否被用過」（時間側信道）→ 要按租戶分隔快取命名空間

> **收尾提醒**：兩邊功能持續趨同（continuous batching、chunked prefill、投機解碼都已經是共識）——**別把某一次 benchmark 當永久結論**。

---

## 問題③ 輸出不可控（第 17–20 頁）

### 四種翻車
多餘文字（「當然！以下是您要的 JSON：」）、型別錯誤（`{"age": "twenty"}`）、語法小錯、幻覺欄位。Agent／function calling 一旦解析失敗整條鏈就斷。
⚠️ **這不需要另一個小模型來審核**——純粹是符號計算／規則引擎問題。

### 共同地基：FSM + 位元遮罩（第 18 頁）

1. **預編譯成 FSM**：JSON schema / 正則 / EBNF → 有限狀態機。**XGrammar 是 SGLang、vLLM、TensorRT-LLM 三家的預設後端**（另可換 Outlines、llguidance）
2. **非同步編譯**：vLLM 讓請求先進 `WAITING_FOR_FSM`，編好才轉 `WAITING`，不阻塞其他請求
3. **位元遮罩**：每步查表得合法集合，其餘 logit 設 −∞，在合法子集重新歸一化採樣 → **物理上不可能生成非法內容**

> ⚠️ **常見誤解**：以為 SGLang 是自研語法引擎、vLLM 是外掛。**實際上兩家的預設後端都是 XGrammar**——差異在下一頁。

### 對比③：jump-forward decoding（第 19 頁）

這是 SGLang「compressed FSM」真正的價值：**不是換一個語法引擎，是在同一個 FSM 上多走幾步。**

觀察：JSON 裡有一大段路徑是**唯一確定**的。`{"name": "` 這 8 個 token 在 schema 確定後就沒有第二種可能 →
- 沒有跳躍：逐 token 推理 8 次
- **jump-forward：直接吐出，0 次推理**，省下的時間拿去服務其他請求

結果：**SGLang 的結構化輸出吞吐約為 vLLM 的 3×**。

> **反直覺但重要**：約束不是成本，是**資訊**——它告訴引擎「這幾個 token 不用問模型」。所以加了語法約束之後，生成反而可能比自由生成更快。

### 問題② 與 ③ 的關係：正交，但有一條紅線（第 20 頁）

| | 前綴複用 | 約束生成 |
|---|---|---|
| 複用／裁剪什麼 | **算力**（避免重複矩陣運算） | **候選空間**（哪些 token 允許被選） |
| 作用在 | 「這段 token 的中間結果」 | 「這個請求走到語法的哪裡」 |
| 判定 | 與請求無關 → **可共享** | 屬於請求本身 → **不可共享** |

⚠️ **紅線**：即使兩個請求共享同一段前綴快取，**它們各自的 FSM 狀態仍必須獨立推進**。

> **通用判準**：可以快取的是「計算結果」，不可以快取的是「請求狀態」。跨機複製 KV 時也要問同一個問題。

---

## 問題④ CPU 成瓶頸（第 21–23 頁）

前面三個解完，每個 forward step 只剩 5–10 ms，而 tokenize、排程決策、取樣、序列化、**數百次 kernel launch** 全在 CPU 上。

### 共同地基：CUDA Graph
初始化時對各種 batch size 做 dummy forward，把整串 kernel launch 錄成 DAG，之後直接 replay。兩家都有（vLLM V1 用 piecewise CUDA graph 兼顧動態形狀）。

### 對比④：兩條路，同一目標（第 22 頁）

| | SGLang：zero-overhead scheduler | vLLM：多進程 + async 排程 |
|---|---|---|
| 做法 | 把 CPU 排程藏進「上一步的 GPU 執行時間」裡，時序上完全重疊 | EngineCore 獨立進程只跑排程＋執行；tokenize／多模態前處理／串流輸出各自重疊 |
| 加上 | v0.4 起導入，是它能撐大規模部署的關鍵 | async scheduling：下一步決策與本步執行重疊 |
| 工程取向 | 單一進程內把時序排好 | 用進程邊界隔開，較易維護 |

**vLLM V1 相對 V0 吞吐提升 ~1.7×，完全來自 CPU 開銷削減——一個 GPU kernel 都沒改。**

> **這一頁值得講**：聽眾很容易以為「推論優化＝寫 CUDA」。實際上一大塊收益來自「別讓 GPU 等 CPU」，跟第一堂 prefetch/overlap 的 demo 是同一個道理。

### 雙胞胎：TTFT vs ITL（第 23 頁）

| 指標 | 被誰決定 | 想要它小，你會想… |
|---|---|---|
| **TTFT** | prefill（compute-bound） | 讓 prefill 立刻插隊、獨佔 GPU |
| **ITL** | decode（memory-bound） | 別讓 prefill 打斷 decode |

衝突點 **head-of-line blocking**：一個 32K prompt 的 prefill 要 0.5–2 秒，這段時間所有串流輸出的使用者都卡住。

**共同解法：chunked prefill**（兩家都有，vLLM V1 預設開啟）——每步固定 token 預算，剩下的預算拿去混 decode。
附帶好處：compute-bound 的 prefill 與 memory-bound 的 decode 混同一批，**兩種瓶頸互補**。
⚠️ 但這只是**緩解**——根治要靠第四堂的 PD 分離。

---

## 單機的天花板（第 24–25 頁，兩家都有）

### 天花板① 投機解碼 / MTP
Draft（n-gram / 小模型 / Medusa / EAGLE / MTP head）→ Verify（一次 forward 跑完 k 個草稿）→ Accept（rejection sampling）。

**為什麼會賺**：權重讀取 = 1 次（不變），FLOPs = k 倍 → **AI 從 1 變成 k** ＝用閒置算力換頻寬。
這是**唯一能在不增加 batch 的前提下改善單請求延遲**的招式 → 低流量、互動式、本機部署特別有效。
- 輸出分佈**嚴格不變**（rejection sampling 保證）
- 期望接受數（接受率 α、草稿長度 k）＝ $(1-\alpha^{k+1})/(1-\alpha)$；α=0.8、k=4 → 約 3.4 個 token/步
- DeepSeek-V3 的 MTP head 報告第二 token 接受率 ~**85–90%**
- **兩家差異**：SGLang 對 DeepSeek 系的 MTP head 支援特別完整；vLLM 內建 n-gram / EAGLE / Medusa 多種可選

### 天花板② 量化
| 對象 | 路徑 | 效果 |
|---|---|---|
| 權重 | fp16 → FP8（÷2）→ FP4/MXFP4（÷4） | decode 步時等比下降 |
| KV cache | fp16 → FP8 → 更低 | KV 變小 ⇒ 放得下更多序列 ⇒ **B 又能更大** |
| Activation | FP8 / MXFP8 | 決定 GEMM 能不能走 FP8/FP4 tensor core |

硬體對齊：Hopper 有 FP8、Blackwell 有 FP4。**精度格式是硬體規格表上的一行**——[第四堂](class4-models-to-racks.md)前半會看到模型端已經開始「出廠就是 4-bit」。

---

## 彙整：四個問題 × 兩種寫法（第 26 頁，本堂核心產出）

| 問題 | 共同地基 | SGLang 的寫法 | vLLM 的寫法 |
|---|---|---|---|
| ① 程式難平行 | — | 前端 DSL：程式＝執行圖 | 沒有對應物（交給應用層） |
| ② KV 怎麼放 | 分頁式 KV、block table | 同（分頁記憶體池） | 同（PagedAttention 命名） |
| ② 怎麼被找到 | 前綴複用 | radix tree，O(D) 最長前綴 | 鏈式雜湊表，O(1) 對齊 block |
| ② 什麼順序進來 | continuous batching | **cache-aware 排程（主動）** | FCFS / priority（被動） |
| ③ 格式保證 | XGrammar FSM + 位元遮罩 | ＋ **jump-forward（約 3× 吞吐）** | ＋ 非同步 FSM 編譯 |
| ④ CPU 開銷 | CUDA Graph | zero-overhead scheduler | 多進程 + async（V1 ~1.7×） |
| ④' TTFT vs ITL | chunked prefill | 有 | V1 預設開啟 |
| 天花板 | 投機解碼、量化 | EAGLE3 / MTP 支援強 | n-gram / EAGLE / Medusa 多選 |

> **看這張表的方法**：**共同地基那一欄是「這個領域已經收斂的共識」**，右邊兩欄才是差異——而**差異幾乎都落在「要不要替使用者猜工作負載」**。

---

## 帶走三句話（第 27 頁）

1. **每個機制都是被一個具體痛點逼出來的。** DSL 因為程式難平行、前綴複用因為程式製造了大量共享前綴、FSM 因為 agent 需要能解析的輸出、CPU 優化因為前三個解完 GPU 快到 CPU 跟不上。**順著問題走，就不用背功能表。**
2. **兩家的差異幾乎都在「要不要替你猜工作負載」。** SGLang 猜你會重複用前綴與語法（radix tree、cache-aware 排程、jump-forward），猜對了就贏很多；vLLM 不猜，把廣度與抽象做好。**共同地基（分頁 KV、continuous batching、XGrammar、CUDA Graph）兩家一致。**
3. **所以選型看流量長相，不看 benchmark 排名。** 前綴共用 >60% → SGLang 的 TTFT 低 20–40%、結構化輸出快約 3×；請求彼此獨立 → 兩者差 <5%，就選硬體與模型支援更廣的那家。

---

## 撐場用：Q&A 速查

**Q：vLLM 和 SGLang 是同一個 level 的工具嗎？**
A：**引擎這一層是，兩者是直接替代品**（都自己管 KV、自己排程、自己執行模型、自己開 OpenAI 相容 endpoint），部署時二選一。但 SGLang 範圍略大：多了**前端 DSL**（vLLM 沒有對應物）與**自帶 router**（vLLM 這層長在生態系裡：production-stack、llm-d、Dynamo）。
不同層的例子：FlashInfer / FlashAttention / DeepGEMM 是 kernel 庫（**被引擎使用**，兩家都會呼叫 FlashInfer）；Dynamo / llm-d / Ray Serve 是編排層（**驅動引擎當 worker**）；llama.cpp / MLX 是本機單人推論，目標函數不同。

**Q：那 TensorRT-LLM 呢？**
A：同一層的第三個選項，但風格偏編譯器——先 build engine 再跑，極致最佳化單一配置，靈活度較低。它的預設語法後端也是 XGrammar。

**Q：MoE 在單卡上不是不省記憶體嗎？稀疏化到底省什麼？**
A：單卡不省**容量**（所有專家都要在 HBM），省的是**每 token 的權重讀取量與 FLOPs**。要連容量也省得靠 EP——第四堂的大規模 EP。

**Q：既然 batch 越大越好，為什麼不無限加大？**
A：①KV 吃光 HBM；②超過 ridge point 後變 compute-bound，步時隨 batch 線性成長 → **ITL 變差**；③尾延遲與公平性。生產上是「在 SLO 之下把 batch 開到最大」的約束最佳化。

**Q：這些跟我只有一張 4090 有關嗎？**
A：非常有關。本機跑模型時 batch 幾乎恆等於 1 → 利用率 ~0.3%，**你買的算力 99% 在閒置**。所以本機端的加速幾乎全部來自天花板那兩條路：**量化**（砍分母）與**投機解碼**（把閒置算力換頻寬）。這也解釋了為什麼 llama.cpp / MLX 的核心賣點永遠是量化格式。

**Q：前綴快取有安全問題嗎？**
A：有值得警惕的方向。快取只在 token 序列相同時命中，但**命中與否會反映在 TTFT 上**，理論上構成時間側信道（可探測「某段 prompt 是否被別人用過」）。多租戶場景通常會按租戶分隔快取命名空間。適合當開放討論。

---

## 資料來源

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（RadixAttention、compressed FSM、前端 DSL、最高 6.4×）
- [RadixAttention 部落格](https://www.lmsys.org/blog/2024-01-17-sglang/)｜[SGLang 結構化輸出文件](https://docs.sglang.io/advanced_features/structured_outputs.html)（XGrammar / Outlines / llguidance 後端）
- [Efficient Memory Management for LLM Serving with PagedAttention（SOSP'23）](https://arxiv.org/abs/2309.06180)（60–80% 浪費 → <4%、2–4× 吞吐）
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)（排程器、KV cache manager、鏈式雜湊 prefix caching、chunked prefill、CUDA Graph、投機解碼、FSM 非同步編譯）
- [vLLM V1 架構升級公告](https://vllm.ai/blog/2025-01-27-v1-alpha-release)（~1.7× 吞吐、統一排程、常數時間淘汰）
- Sarathi-Serve（chunked prefill）、[DeepSeek-V3 技術報告](https://arxiv.org/abs/2412.19437)（MTP 接受率）
- 兩家對照的量化數字（TTFT p50 −37% / p95 −41%、結構化輸出 ~3×）取自第三方評測整理，**工作負載差異極大，開講前建議自己在目標流量上量一次**
