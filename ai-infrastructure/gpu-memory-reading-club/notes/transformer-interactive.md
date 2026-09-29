# 報告 — 玩具級 decoder-only Transformer 互動地圖：全景 → 計算子 matmul（L2/HBM 搬運）、三種執行模式、GPU/TPU/Groq 對應

互動地圖：[../interactive/transformer-map.html](../interactive/transformer-map.html)｜合輯指引頁：第 25 頁（互動環節②）｜TPU 專頁：第 27 頁｜Groq 專頁：第 28 頁（記憶體價格鋪墊頁：第 8 頁）

> **decoder-only 版**：現代 LLM（GPT / Llama / DeepSeek）都是 decoder-only，所以這張地圖聚焦「只有 decoder」的結構——masked self-attention + FFN 疊 N 層、逐字自回歸生成。相對早期版**已移除 encoder、cross-attention 與 memory M**（那是 seq2seq / T5 / Whisper 類的東西，對「LLM 上機」不是主線）。

## 1. 為什麼做一個「玩具級」Transformer

真實模型的數字（d=12288、96 heads、128K context）大到畫不出來也看不懂；但 Transformer 的**結構**——分叉、殘差、attention 的矩陣鏈——跟尺寸無關。所以把所有維度縮到極小但**保留每一個分叉**：

| 維度 | 玩具值 | 真實值（GPT 級） | 縮放比 |
|---|---|---|---|
| 序列長 T | **5**（我 愛 喝 咖 啡） | 128K+ | ~26,000× |
| d_model | **6** | 12,288 | ~2,000× |
| heads | **2** | 96+ | ~48× |
| d_head | **3** | 128 | ~43× |
| FFN hidden | **12**（2×） | 4× d_model | — |

同構意味著：在玩具上看懂的每一條資料流，乘上倍率就是真實模型。

## 2. 地圖的七層（與展示動線）

由宏觀到微觀逐層下鑽，把「Transformer 結構」與「資料在記憶體階層怎麼搬」串成一條線：

1. **全景（decoder-only 自回歸迴圈）**：最宏觀。token 過 embedding + 位置編碼 → 疊 N 層 decoder block（每層 masked self-attn + FFN + Add&Norm）→ logits → 取樣下一個字，**再把生成的字接回輸入、重跑一次**。點 Decoder block 下鑽。GPT 這類只留 decoder，沒有 encoder、沒有 cross-attention。
2. **Block**：一層 decoder block 兩段——masked self-attention（看已生成的自己、擋未來）→ FFN，每段後接殘差 + LayerNorm。attention 是 token 間唯一交換資訊的地方；FFN 對每個 token 獨立（token 軸天然平行）。**多卡時：層內切＝張量平行(TP)，層間切＝管線平行(PP)。**
3. **Attention**：X →（×Wq/Wk/Wv）→ Q、K、V [5×6]，6 維「切」成 2 個 head × 3 維。**分叉＝平行軸**：head 之間零依賴 → GPU 上是 batched GEMM / 不同 tile。MQA / GQA / MLA 就是在這個分叉上動手腳（多個 Q head 共用或壓縮 KV），把 KV cache 變小。
4. **Head**：完整矩陣鏈 Q[5×3]·Kᵀ[3×5] → S[5×5]（causal mask 擋未來）→ softmax → ·V[5×3] → O[5×3]。**這一層是三種模式差異最清楚的地方**（見下節）。點中間的「·」下鑽到計算子。
5. **計算子（一個 matmul × L2⟷HBM）**：把鏡頭推到單一 matmul（S = Q·Kᵀ）。operands 住在大而慢的 HBM，要算時切 tile 沿 **HBM → L2 → shared/暫存器 → tensor core MMA** 流動，結果再寫回 HBM。**這一層專門回答「L2 與 HBM 怎麼交換資料」**：訓練/prefill 是大 GEMM，tile 在 L2/SRAM 重複用、命中率高 → compute-bound（訓練還要把 S、A 留給 backward → 容量壓力）；decode 是 GEMV，operand 用一次就丟、KV cache 又比 L2 大 → 每步都得從 HBM 重搬、箭頭轉紅 → **L2↔HBM 流量＝瓶頸**、tensor core 空轉。
6. **FlashAttention（線上 softmax）**：回答「怎麼讓上一層那個 T×T 的 S 不落 HBM」。上方 6 個步驟 pill（或 `←`/`→`）一步步展開：**① 問題·大小（畫出 S=T×T≈134 MB ≫ 晶片內 SRAM 228 KB，所以只能落 HBM）→ ② safe softmax（減全域 a_max）→ ③ 分塊找 max（block-wise reduction：d₁→d₂→…→d_B=a_max）→ ④ 分塊找 Σ（s_k = s_{k−1}·e^(d_{k−1}−d_k) + Σ e^(a_i−d_k)）→ ⑤ 分塊算 o（o_k = o_{k−1}·(s_{k−1}/s_k)·e^(d_{k−1}−d_k) + Σ(e^(a_i−d_k)/s_k)v_i）→ ⑥ 取代整個 S（一次掃完同時得 a_max/Σ/o，O(T²)→O(T)、快 2–4×、exact）**。這就是「線上 softmax + tiling」的遞迴，把 S 留在 SRAM、取代整個 S 的計算。
7. **硬體**：GPU / TPU / Groq 切換（見第 4、5 節）。

操作：點發亮元件下鑽、`Esc` 回上層、`1`–`7` 跳層、`T`/`P`/`D` 切模式；FlashAttention 層用 `←`/`→` 切步驟；URL hash 可直達（如 `#op.decode`、`#flash`、`#hw.decode.tpu`）。

## 3. 三種執行模式：同一條公式、三種資料流

頂部切換 **訓練 / Prefill / Decode**，所有層的視覺與側欄同步變化：

| | 訓練 | Prefill | Decode |
|---|---|---|---|
| 算的範圍 | 整批 5 列（causal mask） | 同訓練 forward | **只有最後一列** |
| 本步形狀 | Q[5×3]·Kᵀ = GEMM | GEMM | q[1×3]·Kᵀ = **GEMV** |
| K、V 來源 | 現算 | 現算 → **寫入 KV cache** | **從 HBM 讀 KV cache** |
| 額外負擔 | S、A 留給 backward（容量） | — | 每步重讀權重 + KV（頻寬） |
| AI / 判定 | 高 / compute-bound | 高 / compute-bound | ≈1–2 / **memory-bound** |

視覺對應：decode 模式下非最後列全部變暗、K/V 矩陣加上 cyan 外框與「KV cache←HBM」徽章、S 矩陣退化成一條——把合輯 Part 3 的推導變成「看得到」的東西。

**prefill 與 decode 是推論的兩張臉**：prefill 先把整段 prompt 一次算完（大 GEMM、compute-bound、順手寫 KV cache），decode 再逐字吐、每步從 HBM 重讀 KV（GEMV、memory-bound）。訓練的 forward 形狀＝prefill，所以地圖裡「訓練」與「Prefill」共用同一種形狀。

## 4. 對應到 NVIDIA GPGPU 結構（硬體層 · GPU）

硬體層 GPU 視圖畫成一條流水線：**HBM →（串流）→ 晶片內 SRAM →（餵）→ tensor core**。

**(a) 大 GEMM 怎麼變成 tensor core**：右側 tensor core 卡有一個 6×6 大矩陣 + 一個 2×2 高亮 tile 掃過去的動畫 → 「大 GEMM 切成 tile（真實 16×16）→ 每個 tile 一拍餵進 tensor core 做一次 MMA（D = A·B + C）」。這就是 Part 2 tiling 的硬體落地：cuBLAS/cuDNN 把大矩陣乘切塊，逐塊送進 tensor core。

**(b) KV cache 為什麼一直在 HBM↔SRAM 之間搬**（使用者問的重點，已畫進圖）：

- 晶片內 SRAM 很小：L2 ~50 MB、shared/L1 ~228 KB/SM；KV cache 動輒幾百 MB~**GB** → **根本放不進晶片**。
- 所以 KV cache（和權重）住在 **HBM**。每個 decode step，attention kernel 把權重 + KV 從 HBM **串流**過 L2 → shared → register，算完即丟，下一步再重串一遍。
- decode 模式下：串流箭頭變紅 + 「每步重串流」、底部紅色 strip 寫出「KV cache 可達 GB ≫ L2 50 MB ≫ shared 228 KB → 放不進晶片 → 每 token 從 HBM 重新串流 → HBM 頻寬＝速度天花板」、tensor core 利用率掉到 <5%（tile 只剩一條）。
- 訓練 / prefill：權重讀一次、攤給整批 token → tile 餵得滿 → compute-bound、利用率高。
- 連回合輯：Part 1（HBM/SM/tensor core）、Part 2（roofline、tiling→MMA）、Part 3（prefill/decode/KV cache）、Part 5（FlashAttention 讓 S 不落地 HBM、GQA/MLA 讓 KV 變小、DSA 只讀選中的 KV——都是在減這條串流）。

## 4b. TPU（硬體層 · TPU）

weight-stationary 脈動陣列 + XLA 靜態排程的 6×6 動畫（對角波前）。重點：KV **一樣住 HBM**、decode **一樣是 GEMV** → memory-bound 跨硬體成立。詳見第 5 節。

## 5. TPU solution（硬體層 · TPU）

**設計哲學**：GEMM 專用到極致。MXU = 128×128 個 MAC 排成**脈動陣列（systolic array）**，採 **weight-stationary** 資料流——權重先載入、釘在格子裡不動，輸入從左側一拍一拍流入，部分和往下傳、結果從底部流出。**沒有 warp、沒有動態排程器**：XLA 編譯器把整個計算圖先排好，資料跟著 clock 齊步走。地圖中用 6×6 動畫示意（對角線波前掃過 = 計算波）。

**Transformer 在 TPU 上**：

- 訓練 / prefill 的大 GEMM 是甜蜜點：陣列被填滿、利用率高；backward 也是 GEMM。
- **decode 的小 GEMV 一樣餵不滿 128×128 的陣列**——大部分 MAC 空轉，TPU 一樣 memory-bound。**頻寬牆是物理，不是品牌**：換硬體換不掉「每 token 搬一遍權重」這件事。
- bf16 是 TPU 帶進主流的數值格式——「模型 → 硬體 → 模型」迴圈的實例。
- 代價：動態形狀、稀疏、分支不友善（編譯器靜態排程的反面）。

**GPU vs TPU 一句話**：GPU 用「上萬條 lane + 動態 warp 排程」吃吞吐，TPU 用「固定資料流 + 編譯器」吃吞吐——兩條路都是為 GEMM 而生，瓶頸物理相同。

## 5b. Groq solution（硬體層 · Groq）— 直接打 KV cache 的頻寬牆

承上一節「KV cache 住 HBM、每步串流」的痛點，Groq 的設計是把痛點的根源拿掉：

**設計哲學**：**全 SRAM、砍掉 HBM**。Groq LPU（Language Processing Unit）一顆有 ~230 MB 片上 SRAM、**~80 TB/s** 片上頻寬（≈ HBM 的 20×+），完全沒有 HBM/DRAM。排程像 TPU 一樣**全確定性**——compiler 在編譯期把每個操作的時序排好，沒有動態排程、沒有反應式 cache。

**為什麼這對 decode 是關鍵**：decode 是 memory-bound，速度＝「每 token 要重讀的權重 + KV」÷「記憶體頻寬」。Groq 不改分子（一樣要搬），改**分母**——記憶體從 3 TB/s 的 HBM 換成 80 TB/s 的 SRAM → decode token/s 大幅領先。**這正是使用者問題的硬體解**：GPU 因為 KV 塞不下 SRAM 而被迫一直從 HBM 串流；Groq 乾脆讓所有記憶體都是 SRAM。

**代價**：一顆只有 230 MB → 權重 + KV cache 裝不下 → 必須**切片散到幾十～幾百顆 LPU**，用確定性網路串起來。所以 Groq 是「用很多顆晶片 + 海量 SRAM 頻寬」換 decode 延遲。也因此 Groq 目前是**推論專用**（不做訓練；訓練模式下地圖會註記這點）。為什麼一顆只放得起 230 MB？因為片上 SRAM ≈ $5000/GB（合輯第 8 頁「每 GB 價格」）——貴到只能放這麼點，這也是 Groq 必須堆很多顆的根本原因。

**三種硬體一句話**：GPU 在 HBM 上想辦法（FlashAttention 省 S、GQA/MLA 省 KV）、TPU 同樣靠 HBM 但用脈動陣列、Groq 直接把 HBM 換成 SRAM——**記憶體牆是物理，三條路都在繞同一道牆**。

## 6. 建議展示流程（合輯第 25 頁切出來，約 10–14 分鐘）

1. `#io`（訓練模式）：**全景** → token → embedding → 疊 N 層 decoder block → logits → 下一個字 → 「接回輸入、再跑一次」。按 `D` 點出「decode 逐字生成、每步接回輸入」。點 **Decoder block** 下鑽。
2. **Block**：兩段（masked self-attn → FFN）→ 「attention 是 token 間唯一交換資訊處、FFN 對每 token 獨立」。順帶點出「層內切＝TP、層間切＝PP」。點 self-attn 切開。
3. 點開 **Attention**：看 6 維切成 2 head 的分叉 → 「分叉＝平行軸」；提 MQA/GQA/MLA 在此分叉上壓 KV。
4. 點開 **Head**：訓練看 causal mask；按 `P` 講 prefill 寫 KV cache；按 `D` 看整排變暗、只剩一條 GEMV、KV 徽章亮起。點中間「·」下鑽到計算子。
5. **計算子（matmul × L2⟷HBM）**：訓練模式先講大 GEMM「tile 在 L2/SRAM 重複用 → compute-bound、S/A 留給 backward 是容量壓力」；按 `D` → 箭頭轉紅、L2 變「裝不下→miss」、寫回變「算完即丟、下步重載 KV」→ **「L2↔HBM 流量＝瓶頸」**。這是本份報告「L2 與 HBM 怎麼交換」最核心的一頁。
6. 按 `6` 進**硬體層**（GPU）：先指 tensor core 卡的 tile 動畫 →「大 GEMM 切 tile 餵 MMA」；再 D/P 來回切，看串流箭頭變紅、底部 strip 寫「KV 塞不下 → 每步從 HBM 串流」、利用率掉到 <5%。
7. 切 **TPU**：脈動陣列動畫 → 「沒有排程器、資料齊步走」；按 `D` → 「KV 也住 HBM、一樣 memory-bound」。
8. 切 **Groq**：全 SRAM 80 TB/s + 多晶片 → 「把 HBM 換掉就解掉串流；代價是要很多顆」。
9. 切回投影片第 26 頁（模型 → 硬體），第 27 頁 TPU、第 28 頁 Groq 接著講（記憶體價格鋪墊在第 8 頁，可先回放）。

## 7. 檔案與重建

- 互動地圖：`interactive/transformer-map.html`（單檔、離線、無相依；7 層：decoder 全景 → Block（masked self-attn + FFN）→ Attention → Head → 計算子 matmul(L2⟷HBM) → **FlashAttention（線上 softmax 6 步驟）** → 硬體；硬體層三個 tab：GPU / TPU / Groq）。
- 合輯第 25 頁（互動②指引）、第 27 頁（TPU）、第 28 頁（Groq）由 `slides/build/generate_full.js` 產生；記憶體每 GB 價格在第 8 頁。
- 相關講稿：[合輯索引 full-series.md](full-series.md)——prefill/decode/KV cache（Part 3）、Transformer 為平行而生與 FlashAttention/GQA（Part 5）的推導都在其中的「關鍵推導速查」一節。多卡（DP/TP/PP）與 decoder-only 的逐 block × GPU 對應見[第二堂課講稿 class2-transformer-gpu.md](class2-transformer-gpu.md)。
