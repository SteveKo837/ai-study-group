# 第四堂課（最終章）講稿／索引 — 從模型到機櫃：一個字，穿過一整排機櫃

> **狀態：投影片已產出（28 頁）**。2026-09-22 由原第四（SGLang 多機篇）、第五（中國開源模型）、第六（機櫃）三堂合併而成：
> - 原第四堂整堂捨去；它的重點（大規模 EP、PD 分離、cache-aware router、容錯）已在後半段的請求路徑裡逐站出現。
> - 原第五堂精簡成前半段；新增兩頁入門（第 2 頁 decode、第 5 頁 MoE）和一頁接縫（第 12 頁 DeepSeek-V3 規格），各實驗室的細節頁併進全景表（第 10 頁）。
> - 原第六堂是後半段，**拿掉 NVIDIA vs AMD 三頁**，互動教具也拿掉了 MI355X。
>
> 投影片 [../slides/class4-models-to-racks.html](../slides/class4-models-to-racks.html)（由 `slides/build/generate_class4.js` 產生：`cd ../slides/build && node generate_class4.js`）。互動教具 [../interactive/rack-journey-map.html](../interactive/rack-journey-map.html)（第 24 頁指引）。
> 前置：[第一堂](full-series.md)（roofline、記憶體階層、decode 為何 memory-bound）、[第二堂](class2-transformer-gpu.md)（EP 與互連頻寬階梯）、[第三堂](class3-engine-single-node.md)（AI ≈ B、分頁 KV、prefix caching、CUDA Graph）。

---

## 0. 這一堂要回答什麼

**聽眾的原始問題**：當運行單位是數個機櫃、數十個 node 時，SGLang / vLLM 怎麼把 DeepSeek 這種 671B MoE 跑完「一輪」inference？

**為什麼前半要先講模型**：後半的每一站都假設聽眾知道「decode 一次只吐一個字」「MoE 每個字只挑 8 個專家、要把字送到專家所在的卡」「MLA 讓 KV 很小」。前三堂只講到單機引擎，沒碰過 MoE 的內部，也沒談過中國開源模型為什麼這樣設計。所以前半先把這些零件講清楚，後半才把它們裝回機櫃。

**接上一堂的口白**：「第三堂是框架從外面調——把 batch 撐大。今天先看模型從裡面改：DeepSeek 這批模型怎麼讓每個字少搬一點；再把這個模型搬上 12 台機器，跟著一個請求從進門走到吐出最後一個字。」

| 項目 | 內容 |
|---|---|
| **鉤子** | DeepSeek 尖峰開 278 台 8 卡機服務 V3/R1；SGLang 用 12 台開源復現。你螢幕上每跳出一個字，decode 池的 72 張卡要一起完成 116 次全員交換。**而且拆到越多卡上，每張卡的產量反而越高。** |
| **鉤子拋出的問題** | 通訊明明變多了，為什麼拆散反而便宜？ |
| **主軸（≤30 字）** | **拆散變便宜的前提：最常搬的資料，走最快的路。** |
| **軸型** | 前半是「零件軸」：decode → 五個旋鈕（MLA、MoE、少看、MTP、降精度）→ 接縫頁把規格對到機櫃上的每一站。後半是拆解軸：按「多常搬 × 一次多大」把資料分四種，中段照請求時序走，合回來指出真瓶頸＝**一個 scale-up 域裝得下幾張卡**。 |
| **閉環** | 結尾回答鉤子：72 張卡同步踏一步之所以划算，是因為每字 116 次的 MoE 交換被 DeepEP 壓進 µs 級的特化 RDMA，而一次性的 KV 交接小到（MLA）可以走慢車道；再往上的 4.4×，買的是更大的 NVLink 域。 |

### 三種角色的案例分工

| 角色 | 用哪一套 | 為什麼 |
|---|---|---|
| 鉤子的規模感 | DeepSeek 官方（H800、尖峰 278 台、decode EP144） | 真實線上流量與成本，只出現在第 1、13 頁 |
| **跟著走的主角** | **SGLang 96×H100**（prefill 4 台 EP32、decode 9 台 EP72，在同一個 12 台叢集上分開量測） | 開源框架、可復現、每個手法都有單獨量過的增益 |
| 對照組 | B200 8 卡機、GB200 NVL72 | 只在第 24–26 頁與互動教具出現 |

### 閘門一自檢

| 題 | 結果 |
|---|---|
| 主軸 30 字內說得出來？ | ✅「拆散變便宜的前提：最常搬的資料，走最快的路」（20 字） |
| 標題鏈讀起來像摘要？ | ✅ 見 §1 |
| 每頁拿掉主軸會斷？ | ✅ 前半每頁都在某一站被後半回收（第 12 頁的對照表明寫出回收點）；第 16–23 頁是請求路徑，順序不可換 |
| 相鄰兩頁能互換？ | ✅ 第 2 頁（decode）必須在旋鈕之前；第 5 頁（MoE 機制）必須在第 6 頁（稀疏度）之前；第 12 頁是前後半的接縫；24（教具看現象）必須在 25（解釋）之前 |

---

## 1. 頁面地圖＝標題鏈（28 頁）

> **快測**：只讀這一欄，應該就是一篇完整摘要。

| 頁 | 標題（斷言句） | 來源 | 本頁回答 → 拋出 |
|---|---|---|---|
| **鉤子** | | | |
| 1 | 你看到的每一個字，背後有 72 張卡同時踏了一步 | 原六-1 | → 為什麼要這麼多張卡一起踏？先認識這個模型 |
| **前半：先認識要搬上機櫃的模型** | | | |
| 2 | 一個字是怎麼生出來的：prefill 一次吃完 prompt，decode 一次只吐一個字 | **新增** | decode 每字要搬權重＋KV → 模型端能怎麼讓它少搬？ |
| 3 | 中國開源模型的思路：五個旋鈕，讓每個字少搬一點 | 原五-3＋4 | 全景 → 逐一看 |
| 4 | 旋鈕①：壓 KV —— MHA → GQA → MLA | 原五-5 | KV 變小 → FFN 那一半呢？ |
| 5 | MoE：把一個大 FFN 換成 256 個小 FFN，每個字只挑 8 個走 | **新增** | MoE 的動作與 dispatch／combine → 稀疏度推到多高？ |
| 6 | 旋鈕②：少算 —— MoE 稀疏度一路往上推 | 原五-6 | 不省容量 → 長 context 呢？ |
| 7 | 旋鈕③：少看 —— 稀疏 vs 線性，兩條不同的路 | 原五-7 | 兩條路 → 為什麼共識是混合＋稀疏？ |
| 8 | 最有價值的反例：MiniMax M1 → M2 → M3 | 原五-8＋9 | 理論更省 ≠ 實際更快 → 剩下兩個旋鈕 |
| 9 | 旋鈕④ 一次多產、旋鈕⑤ 降精度 | 原五-10 | → 五家各轉了哪些？ |
| 10 | 全景：五個實驗室 × 五個旋鈕 | 原五-14（吸收 11–13） | → 模型改了，框架跑得動嗎？ |
| 11 | 為什麼他們連 kernel 都開源？ | 原五-15 | 四個零件 → 它們會在機櫃上出現 |
| 12 | 等一下要搬上機櫃的，就是這個模型：DeepSeek-V3 | **新增** | 規格 × 機櫃上的每一站 → 拆散為什麼反而便宜？ |
| **後半：拆法** | | | |
| 13 | 把 DeepSeek 拆到越多卡上，每張卡反而產出越多 token | 原六-2 | 拆散真的更便宜 → 通訊變多了，怎麼會？ |
| 14 | 一個「模型服務」不是一台機器，是一個 router、兩個池子、三張網 | 原六-3 | 服務長什麼樣 → 請求在裡面搬了哪些東西？ |
| 15 | 機櫃群＝多了三層的記憶體階層：每種資料走它付得起的路 | 原六-4 | 拆法 → 從進門那一跳開始走 |
| **後半：跟著一個請求走** | | | |
| 16 | 進門那一跳只搬幾 KB，卻決定了一半的 prefill 要不要算 | 原六-5 | router 選誰很重要 → prefill 怎麼跨機算？ |
| 17 | Prefill 是整批的大塊通訊，跨 InfiniBand 也吃得消 | 原六-6 | prefill 可以跨機 → KV 怎麼交給 decode？ |
| 18 | KV 交接不是「傳過去」而已：SGLang 先預留再推，vLLM 先算完再拉 | 原六-7 | 交接流程 → 搬這一包要多久？ |
| 19 | 一份 5K token 的 KV 跨機只要 7 ms——MLA 在前半段就替這一跳付了帳 | 原六-8 | KV 交接不是瓶頸 → decode 一步在忙什麼？ |
| 20 | Decode 一步：attention 各算各的，MoE 大家一起交換 | 原六-9 | 熱迴圈長相 → 這 116 次要花多少時間？ |
| 21 | 不做重疊時，通訊佔掉 decode 一步約四成 | 原六-10 | 通訊很貴 → 那為什麼拆散還是比較便宜？ |
| 22 | 拆散之所以便宜：每卡只放 4 個專家，省下的 HBM 拿去開大 batch | 原六-11 | 回答鉤子的一半 → 代價是什麼？ |
| 23 | 代價是 72 張卡綁成一個節拍器：最慢那張決定速度 | 原六-12 | 代價 → 什麼條件下付得起？ |
| **後半：合回來** | | | |
| 24 | 互動環節：同一個請求放到三種硬體上各走一次 | 原六-13 | 看到現象 → 為什麼只有那一站？ |
| 25 | 四種資料只有一種付不起跨機的路 | 原六-14 | 真瓶頸 → 有實測證據嗎？ |
| 26 | 同一顆 GPU、同樣的 NVLink 頻寬，只把域從 8 擴到 72 | 原六-15 | 證據 → 整個系列在這趟旅程裡的位置 |
| **收尾** | | | |
| 27 | 這趟旅程的每一站，都能在前面找到它的那一頁 | 原六-19（改寫） | 全系列收束 |
| 28 | 帶走三句話 | 原五-16＋原六-20 合併 | 回答第 1 頁 |

---

## 2. 前半逐頁：先認識要搬上機櫃的模型（第 1–12 頁）

> 強度欄：**可查證**＝一手來源；**單一來源**＝廠商自報、只有一家；**推算**＝本講稿自己算的，標註假設。
> 頻寬一律寫**每方向**；NVIDIA 規格頁的 900 GB/s（NVLink 4）、1.8 TB/s（NVLink 5）是雙向合計，引用時會註明。

### 第 1 頁 · 鉤子（封面）

- DeepSeek 在 2025-02-27～28 的 24 小時裡：輸入 **608B** token（其中 **342B / 56.3%** 命中磁碟 KV 快取）、輸出 **168B** token；尖峰佔用 **278 個 node**、平均 **226.75 個**，每 node 8 張 H800；decode 部署單元＝18 node、EP144。〔可查證：DeepSeek open-infra-index day 6〕
- SGLang 在 **12 台 × 8 張 H100** 上開源復現：prefill 4 台（EP32）、decode 9 台（EP72）。〔單一來源：LMSYS 2025-05-05〕
- 「116 次」＝ DeepSeek-V3 共 61 層，前 3 層是 dense，**58 層 MoE × 每層 dispatch + combine 各一次**。〔可查證：config.json `first_k_dense_replace=3`〕

**講法**：「DeepSeek 自己開了 278 台機器在服務。今天我們不看它的黑盒子，而是看開源的 SGLang 怎麼用 12 台做出同一套架構。你手機上每跳出一個字，decode 那 72 張卡就同時做完 116 次全員交換。聽起來很浪費——下一頁會看到，這是最便宜的做法。」

### 第 2 頁 · 一個字是怎麼生出來的（新增）

這一頁補的是前三堂默認、後半卻大量使用的前提。

- **兩個階段**：
  - **Prefill（一次）**：整段 prompt 的 T 個 token 同時進模型 → 大矩陣乘法、算術強度高 → compute-bound。副產品是每一層的 K、V，存起來就是 KV cache。
  - **Decode（每個字一次）**：每步只餵上一個字，產出下一個字。要讀一次整份（活躍）權重和這條序列的全部 KV，卻只算 1 個 token → AI ≈ 1 → memory-bound。
- **自回歸**：字 2 要等字 1 出來才能開始算——decode 無法在時間軸上平行，只能靠「同時服務很多條序列」（batch）把權重讀取攤掉（第三堂 AI ≈ B）。
- **一層 Transformer 只做兩件事**：
  - Attention 回頭看過去所有字 → 讀 KV cache → 旋鈕①壓 KV、③少看動的是這裡。
  - FFN 讓每個字各自過一個大 MLP → 讀權重（佔參數大半）→ 旋鈕② MoE 動的是這裡。

> **講法**：「第一堂我們說 batch=1 的 decode 算力用不到 5%。今天換個角度：一個字是怎麼吐出來的？prompt 一次吃進去，這叫 prefill；之後每吐一個字，都要把整個模型和這段對話的記憶（KV）從 HBM 讀一遍——這叫 decode。整堂課只追一個問題：每吐一個字，要搬多少 bytes、從哪搬到哪。」

### 第 3 頁 · 五個旋鈕（原第五堂第 3、4 頁合併）

出口管制下算力受限，**效率不是加分項，是生存條件**。這批實驗室的三個共同特徵：架構層就為推論成本設計（MLA 的發明動機就是 HBM 頻寬帳單）、系統零件跟著開源（第 11 頁）、技術報告寫得很細、含失敗嘗試（第 8 頁 MiniMax）。

框架從外面調＝把 batch 撐大（動 roofline 的分子，第三堂）；模型從裡面改＝讓每產一個字要搬的 bytes 變少（動分母）。
| 旋鈕 | 打擊的瓶頸 | 代表技術 |
|---|---|---|
| **① 壓 KV** | decode 讀 KV 的頻寬 + 容量 | MHA → GQA → **MLA**（低秩 latent）→ Gated MLA |
| **② 少算** | 每 token 的 FLOPs 與權重讀取 | **MoE 稀疏化**：細粒度專家、共享專家、極高稀疏比 |
| **③ 少看** | 長 context 的 O(n²) 與 KV 線性增長 | **稀疏注意力**（DSA/CSA/MSA）、**線性注意力**（Lightning/GDN/KDA）、混合層 |
| **④ 一次多產** | 單請求延遲（memory-bound 天花板） | **MTP** ＋ 投機解碼 |
| **⑤ 降精度** | 搬的位元組數 | FP8 訓練、**MXFP4 權重 / MXFP8 activation 的 QAT** |

> 每個旋鈕都在回答同一個問題：**怎麼讓每產一個 token，少搬一點位元組？** 這正是第一堂 roofline 的分母。

### 第 4 頁 · 旋鈕① 壓 KV

| | 做法 | KV 每 token |
|---|---|---|
| **MHA** | 每個 head 各存 K/V | Llama 式 32 heads：**512 KB** |
| **GQA** | 多個 query head 共用一組 K/V | Llama-3-8B（8 KV heads）：**128 KB**（÷4） |
| **MLA** | K/V 投影成低秩 latent 再存，用時解回 | DeepSeek-V3：**≈ 70 KB**（576 維 latent × 61 層 × 2B；同規模 MHA 推算 ~4 MB（3.8 MiB），與第 19 頁同一個數） |

DeepSeek-V2 論文自陳：MLA 讓 KV cache 相對 MHA **減少 93.3%**。Kimi K3 用 Gated MLA、GLM-5 也採用 MLA——**這個旋鈕已經是共識**。

> **最該記的一句**：MLA 是「為了 decode 的 HBM 頻寬」而發明的注意力機制——**架構決策就是硬體帳單**。

### 第 5 頁 · MoE：把一個大 FFN 換成 256 個小 FFN（新增）

MoE（Mixture of Experts）只改一層裡的 FFN，attention 照舊。

| 步驟 | DeepSeek-V3 的做法 |
|---|---|
| ① gate 打分 | 一個小線性層替 256 個 routed 專家各打一個分數 |
| ② 挑 top-k | 取分數最高的 **8 個**（實際還有 group-limited routing：8 組先挑 4 組） |
| ③ 專家計算 | 8 個專家各自是一個小 FFN，各算一份 |
| ④ 合併 | 依 gate 分數加權相加，再加上 **1 個 shared 專家**（每個字必經）的輸出 |

- **省什麼**：每個字只讀 8＋1 個專家的權重、只做那麼多 FLOPs → 671B 總參數裡，每個字只動到 37B。
- **不省什麼**：256 個專家全部要放在 HBM（容量）；而且專家一多，一張卡放不下，只能散到很多卡上（EP，專家平行，第二堂 Part B）。
- **散到多卡之後多出來的通訊**：
  - **dispatch**：把這個字的 hidden state 送到它選中的專家所在的卡
  - **combine**：專家算完，把結果送回這個字所在的卡
  - 每一層 MoE 都要來回一次，而且所有卡同時互相送 → all-to-all。DeepSeek-V3 有 58 層 MoE → **每個字 116 次交換**（第 1 頁的數字就是這樣來的）。

> **講法**：「MoE 像一間有 256 位專科醫師的醫院，每個病人只看其中 8 位。看診費（計算）省了，但 256 位醫師都得有診間（HBM），而且診間分散在很多棟大樓——病人得被送過去、看完再送回來。今天後半段的主角，就是這趟接送。」

### 第 6 頁 · 旋鈕② 少算

決定 decode 速度的是**活躍參數 + KV**，不是總參數。所以趨勢是「總參數大幅變大、活躍參數只小幅變大」。

| 模型 | 總參數 / 活躍 | 專家配置 | 活躍比例 |
|---|---|---|---|
| DeepSeek-V3 | 671B / 37B | 256 routed + 1 shared，選 8 | 5.5% |
| Kimi K2 | 1T / 32B | 384 專家，選 8 | 3.2% |
| Kimi K3 | 2.8T / 104B | 896 專家，選 16（Stable LatentMoE） | 3.7% |
| Qwen 3.5 | 397B / 17B | 極高稀疏 | 4.3% |
| GLM-5 | 744B / 40B | — | 5.4% |

- **細粒度專家**：切更小、選更多 → 組合數變多，表達力上升
- **共享專家**：每 token 必經，承接共通知識 → 讓 routed 專家專心學差異
- **無輔助損失負載均衡**（DeepSeek）：傳統 aux loss 逼均衡會傷品質 → 改成動態調整路由 bias


⚠️ **但 MoE 在單卡上不省容量**（專家都得在 HBM），只省每 token 的 FLOPs 與權重讀取。**要連容量也省，得靠後半段的大規模 EP（第 22 頁）**——代價是 all-to-all，以及專家負載不均（＝資料傾斜，第 23 頁）。

### 第 7 頁 · 旋鈕③ 少看：稀疏 vs 線性（前半最重要的分類）

| | 稀疏注意力 | 線性注意力 |
|---|---|---|
| 做法 | **保留完整 KV**，但每個 query 只看 top-k 個位置 | **不存 KV**，改成固定大小的遞迴狀態 |
| 代表 | DeepSeek DSA / CSA+HCA、MiniMax MSA、GLM-5 | MiniMax Lightning、Qwen Gated DeltaNet、Kimi KDA |
| 生態影響 | **KV 還在 → prefix caching / 投機解碼還能用** | **KV 沒了 → 三個生產系統全部要重做** |

> **這個分類是理解 2026 的鑰匙**：共識不是「線性取代 full attention」，而是「**混合 + 稀疏**」——因為稀疏保留了「KV 還在」這個前提。

### 第 8 頁 · MiniMax：最有價值的反例（原第五堂第 8–9 頁合併）

| 版本 | 做法 | 結果 |
|---|---|---|
| **M1**（2025-06） | Lightning Attention 混合（線性為主）、CISPO RL | 宣稱長生成場景 FLOPs 大幅低於同級 |
| **M2**（2025-10） | **退回 full attention** | 品質優先；並公開說明為什麼 |
| **M3**（2026） | **MSA**（MiniMax Sparse Attention）——改走稀疏 | 宣稱 1M ctx 下 prefill **9×**、decode **15×** 快於 M2 |

#### 「No Free Lunch」的三個理由

1. **評測會騙人**：混合注意力在 MMLU / BBH / LongBench 上看起來沒問題，**放大後才發現多跳推理明顯退化**——把散落在長文件裡的線索串起來的能力壞掉了。而要在困難任務上得到統計顯著訊號，所需算力是天文數字。
   > **弔詭**：「省算力的方法」，需要巨量算力才驗證得了。這是效率研究最大的結構性障礙。
2. **理論 FLOPs ≠ wall-clock**：線性注意力的實作**本身就是 memory-bound**，即使在訓練時也吃不滿算力——完全是第一堂 roofline 的教訓：省下的是紙上的 FLOPs，不是牆上的時間。
3. **卡在第三堂建好的三個生產系統上**：

| 系統 | 對應第三堂 | 打壞了什麼 |
|---|---|---|
| **KV cache** | 問題② | 線性狀態對數值精度遠比 full attention 敏感 → 低精度存不了，**旋鈕⑤ 跟著失效** |
| **Prefix caching** | 問題② | 線性狀態不像 KV 可以直接切片複用 → **RadixAttention 的整套價值歸零** |
| **投機解碼** | 天花板① | 在線性骨幹上「仍是未解問題」→ **單請求延遲的唯一解法沒了** |

他們也試過滑動窗口混合，調過比例、RoPE 設定、層內/層間配置、sink token——**在 agent 任務與複雜長文評測上一致地很差**。

> **帶走**：「理論複雜度更低」離「生產環境更快」隔著三層：**kernel 效率、評測有效性、生態相容性**。
> 這也順帶回答聽眾常問的「Mamba／線性注意力不是早就贏了嗎？」——沒有，而且原因非常具體。

### 第 9 頁 · 旋鈕④⑤

#### ④ MTP（Multi-Token Prediction）

訓練時多預測幾步當額外訊號（更密的監督），推論時那些 head **直接當投機解碼的 draft**。

驗證 k 個草稿 token：權重讀取 = 1 次（不變），FLOPs = k 倍 → **AI 從 1 變成 k**（第三堂天花板①的同一把尺）。

DeepSeek-V3 報告第二 token 接受率 ~**85–90%**；Qwen3-Next 也內建 MTP。
> 這是模型端**主動配合投機解碼**的做法——訓練時就把 draft 模型長在自己身上。

#### ⑤ 降精度：從「部署後處理」變成「訓練的一部分」

| 階段 | 做法 | 意義 |
|---|---|---|
| 以前 | 訓練用 bf16 → 社群事後量化成 GGUF/AWQ | 品質掉多少看運氣 |
| **DeepSeek-V3** | **FP8 訓練**（首個大規模開源前沿模型） | 細粒度 scaling + 高精度累加解決數值問題 |
| **Kimi K3** | **MXFP4 權重 / MXFP8 activation，從 SFT 起 QAT** | **出廠就是 4-bit**，直接對齊 Blackwell FP4 |

### 第 10 頁 · 全景：五個實驗室 × 五個旋鈕

| | ① 壓 KV | ② 少算（MoE） | ③ 少看 | ④ 一次多產 | ⑤ 降精度 |
|---|---|---|---|---|---|
| **DeepSeek** | MLA（−93.3%） | 671B-A37B 細粒度+共享 | DSA → CSA+HCA（1M：FLOPs 27%、KV 10%） | MTP 85–90% | FP8 訓練 |
| **Kimi** | MLA → Gated MLA | 1T-A32B → 2.8T-A104B | KDA 線性 ×69 + full ×24 | — | MXFP4 QAT |
| **MiniMax** | — | MoE | Lightning → 退回 full → MSA | — | — |
| **Qwen** | GQA | 397B-A17B | Gated DeltaNet : full ＝ 3:1 | MTP | — |
| **GLM** | MLA | 744B-A40B | DSA 式稀疏 | — | — |

> ⚠️ 以各家技術報告／官方部落格公開數字為準；2026 上半年版本迭代極快，**開講前請對一次官方頁面**。

#### 各實驗室細節（原第五堂第 11–13 頁，未上投影片，Q&A 備查）

##### DeepSeek

| 技術 | 旋鈕 | 重點 |
|---|---|---|
| MLA | ① | V2 自陳 KV cache 相對 MHA 減少 93.3%；V3 ≈ 70 KB/token |
| DeepSeekMoE | ② | 細粒度 + 共享專家；V3：671B-A37B |
| 無輔助損失均衡 | ② | 動態調整路由 bias，均衡且不干擾主目標 |
| MTP | ④ | 第二 token 接受率 ~85–90% |
| FP8 訓練 | ⑤ | 首個大規模用 FP8 完成訓練的開源前沿模型 |
| DualPipe + 通訊 kernel | 系統 | all-to-all 與計算重疊（第 21 頁的 two-batch overlap 是同一類手法） |
| **DSA**（V3.2-Exp, 2025-09） | ③ | 細粒度稀疏注意力 → API 降價 >50%（$0.27/M input） |
| **V4**（2026-04 預覽） | ③ + 系統 | CSA + HCA 逐層交錯、mHC 殘差、Muon。1.6T-A49B / 284B-A13B，1M ctx；**1M ctx 下 FLOPs 只需 V3.2 的 27%、KV cache 只需 10%** |

##### Kimi

| | K2（2025-07） | K3（2026-07） |
|---|---|---|
| 規模 | 1T / 32B 活躍 | **2.8T / 104B 活躍** |
| MoE | 384 專家 / 選 8 | **896 專家 / 選 16**（Stable LatentMoE，latent 3584） |
| 注意力 | MLA | **93 層 ＝ 69 KDA ＋ 24 Gated MLA** |
| 訓練 | **MuonClip**（Muon + QK-Clip）；15.5T tokens 無 loss spike | 宣稱 scaling efficiency ≈ **2.5× K2** |
| 精度 | — | **MXFP4 / MXFP8 QAT** |
| 其他 | — | 原生多模態（401M vision encoder）、1M ctx |

##### Qwen / GLM

- **Qwen3**（2025）：GQA + MoE（235B-A22B 等）
- **Qwen3-Next**：Gated DeltaNet : full ＝ **3:1**，加 MTP，極高稀疏（80B-A3B）
- **Qwen 3.5**（2026-02）：397B-A17B，延續 GDN 3:1；宣稱 256K ctx decode 比 Qwen3-Max 快 **19×**
- **GLM-5**（2026-02）：744B-A40B，**MLA ＋ DSA 式稀疏同時用上**，200K ctx

> **注意「混合比例」已經變成一個新的超參數**：Qwen 3:1、Kimi K3 約 3:1（69:24）——大家收斂到差不多的比例，這本身就是訊號。

### 第 11 頁 · 為什麼他們連 kernel 都開源？（前後半的接縫）

**困境**：新架構如果沒有 kernel，就沒有人跑得動。MLA 不是標準 attention，vLLM / SGLang 原本的 FlashAttention kernel 直接用不了；MoE 的 all-to-all、FP8 GEMM、專家負載均衡也一樣。**開源模型權重卻沒有配套 kernel，等於發布了一台沒有輪子的車。**

| 零件 | 是什麼 | 讓哪個旋鈕真的跑得動 |
|---|---|---|
| **FlashMLA** | MLA 的 decode kernel | ① 壓 KV |
| **DeepEP** | 專家平行的 all-to-all 通訊庫 | ② 少算（在多機可行；第 17、20 頁） |
| **DeepGEMM** | FP8 GEMM | ⑤ 降精度（吃到 tensor core） |
| **EPLB** | 專家平行負載均衡器 | 專家熱點＝資料傾斜（第 22–23 頁） |

> **開源 kernel 是讓自家架構進入 vLLM / SGLang 生態的手段——模型與框架是共生的，不是上下游。**
> 四個零件等一下會在機櫃上一一出現：prefill 用 DeepEP normal 模式、decode 用 low-latency 模式，專家用 DeepGEMM 算，EPLB 決定熱門專家放幾份。

### 第 12 頁 · 等一下要搬上機櫃的，就是這個模型：DeepSeek-V3（新增，前後半的接縫）

| 規格 | 數字 | 前半段哪一頁 | 到了機櫃上變成… |
|---|---|---|---|
| 層數 | 61 層：前 3 層 dense、58 層 MoE | 第 5 頁 | 每個字 58 × 2 ＝ **116 次**交換（第 20 頁） |
| 專家 | 256 routed + 1 shared，每字選 8 | 第 5–6 頁 | decode 72 張卡，每卡只放 **4 個**（第 22 頁） |
| 參數 | 671B 總 / 37B 活躍；hidden 7168 | 旋鈕② | 權重 **688.6 GB**，一台 8×H100 放不下（第 14 頁） |
| 注意力 | MLA，≈ 70 KB / token | 旋鈕① | 一份 5K token 的 KV 跨機只要 **7 ms**（第 19 頁） |
| 精度 | FP8 訓練與權重 | 旋鈕⑤ | dispatch 用 FP8、combine 用 BF16（第 20 頁） |

〔可查證：DeepSeek-V3 config.json（`num_hidden_layers=61`、`first_k_dense_replace=3`、`n_routed_experts=256`、`num_experts_per_tok=8`、`hidden_size=7168`、`kv_lora_rank=512`、`qk_rope_head_dim=64`）；權重大小見 HF repo〕

> **講法**：「前半段講的每個設計，到了機櫃上都會變成一筆搬運帳。671B 一台機器放不下，只能拆；但拆得越散、卡之間的交換越多——照常識應該越慢越貴。下一頁的實測卻是反的。」

---

## 3. 後半逐頁：跟著一個字穿過機櫃（第 13–26 頁）

### 第 13 頁 · 拆散反而更便宜（被誤解的常識）

常識：「模型拆越散，通訊越多，一定越慢越貴。」反例：

| 對照 | 每張 GPU 的產出 | 來源 |
|---|---|---|
| SGLang：PD 分離 + 大規模 EP vs TP16 | decode **5.2×**、prefill **3.3×** | LMSYS 2025-05-05〔單一來源〕 |
| TRT-LLM：EP16/32 vs EP4/8 | 每卡輸出**最高 6.17×**（含 MTP） | NVIDIA TRT-LLM tech blog〔單一來源〕 |
| InferenceX：GB200（decode EP16）vs B200（decode EP8），R1 FP4、125 tok/s/user | **4,130 vs 941 tok/s/GPU（4.4×）** | SemiAnalysis 2026-05-23〔可查證，第三方〕 |

> ⚠️ 三組條件不同，不能互相換算——共同點是方向一致：**拆散（EP 變大）→ 每卡產出變多。**

### 第 14 頁 · 服務長什麼樣（V1 拓樸圖）

圖要畫出（由外到內）：

1. **前端網路**（資料中心乙太）：使用者 → router
2. **Prefill 池**：4 台 × 8 張 H100，EP32
3. **Decode 池**：9 台 × 8 張 H100，EP72——兩個池子是在同一個 12 台叢集上**分開量測**的
4. **三張網**：機內 NVLink 4（每方向 450 GB/s、8 卡一域）／跨機 RDMA（每卡一張 400G ≈ 50 GB/s）／前端乙太（ms 級）
5. 儲存（權重 688.6 GB、磁碟 KV 快取）掛在旁邊

原子：
- DGX H100：8×80 GB、NVLink 900 GB/s（雙向）、10×ConnectX-7 400G（8 張算力網＋2 張儲存網）。〔可查證：NVIDIA DGX H100 頁〕→ LMSYS 原文**未載明網卡配置**，本堂以此典型配置估。
- DeepSeek-V3 權重在 HF 上 163 個 safetensors、**688.6 GB（641.3 GiB）**，其中 680.6B 參數是 FP8。〔可查證：HF repo〕→ **8×80 GB 的一台 H100 連權重都放不下。**

### 第 15 頁 · 本堂地圖：四種資料 × 四條路

| 資料 | 一次多大 | 多常搬 | 付得起的路 | 對應頁 |
|---|---|---|---|---|
| 請求本身（token ids、串流回傳） | KB | 每請求一次 / 每字一次 | 前端乙太（ms 級）就夠 | 16 |
| **KV cache**（prefill → decode） | ≈ 351 MB（MLA BF16，4,989 token） | 每請求一次 | 跨機 RDMA（~50 GB/s） | 18–19 |
| **MoE hidden states**（dispatch / combine） | 每卡每層 ≈ 14 MB ＋ 28 MB | **每層 × 每步 × 72 張卡同步**，每字 116 次 | 只有 scale-up 域或特化 RDMA 付得起 | 20–25 |
| 權重 | 688.6 GB 全部；單專家 FP8 ≈ 42 MiB | 幾乎不搬（啟動、EPLB 重排時） | 儲存網 / RDMA | 23 |

〔MoE 那一列為推算：每卡 256 序列 × 平均送往 7.5 張卡 × hidden 7168，dispatch FP8、combine BF16〕

**講法**：「第一堂那張記憶體階層表，暫存器 → L1 → L2 → HBM → PCIe → SSD，每往外一層慢一個數量級。機櫃群只是**再往外加三層**：scale-up 域 → 跨機 RDMA → 前端乙太。推論引擎的全部工作，就是讓每一種資料只走它付得起的那一段路。」

### 第 16 頁 · 第一站：router

- **SGLang PD 模式**：router（`sglang_router --pd-disaggregation`）挑一組 prefill／decode，塞入 `bootstrap_host / bootstrap_port / bootstrap_room`（隨機 63-bit ID），**同時 POST 給兩邊**；串流從 decode 回來。〔可查證：sgl-model-gateway `mini_lb.py`〕
- **vLLM / llm-d**：Gateway → Endpoint Picker（EPP，KV-aware 排程）→ sidecar → prefill pod → NIXL → decode pod。〔可查證：llm-d docs v0.9〕
- **NVIDIA Dynamo**：Smart Router（KV-aware），Dynamo 1.0 於 2026-03-16 發布，整合 SGLang / vLLM / llm-d / LMCache。〔可查證〕
- **為什麼這一跳重要**：DeepSeek 線上 **56.3%** 輸入 token 命中快取——router 送錯地方，這一半就得重算。（兩難：越追快取命中，請求越容易擠在少數機器上——局部性 vs 負載均衡）

### 第 17 頁 · 第二站：prefill

- DeepSeek-V3 prefill 是 compute-bound（第一堂：大 GEMM、AI 高）；通訊是「一批 token 一次送」的大塊流量。
- **DeepEP normal 模式**（prefill 用）：機內 NVLink 轉發 **153 GB/s**、跨機 RDMA **43–58 GB/s**；不支援 CUDA Graph。〔可查證：DeepEP v1.2.1 README〕
- **Two-batch overlap**：把一批切成兩個 micro-batch，一個在通訊時另一個在算 → prefill 吞吐 **+27–35%**。〔單一來源：LMSYS〕
- 數字錨：prefill 4 台（EP32）在 1K / 2K / 4K 輸入下分別 **57,674 / 54,543 / 50,302 tok/s／node**。〔單一來源〕
- EPLB 共 288 個專家槽（256 + 32 冗餘）→ prefill 每張卡放 **9 個 routed 專家**（288 ÷ 32）。〔可查證：LMSYS〕

### 第 18 頁 · 第三站：KV 交接的流程（V3 三泳道）

> 🎛 **教具對應**：第 2 層「KV 交接」，上方可切 SGLang / vLLM 逐步播放。

**SGLang——並行送、先預留、再推**（源碼 main branch）：

1. router 同時送請求給 prefill 與 decode，帶同一個 `bootstrap_room`
2. decode 查 prefill 端的 bootstrap server → 握手（Bootstrapping）
3. decode **預先配置 KV 槽位**（PreallocQueue）
4. prefill 算 forward（Waiting → Inflight，非阻塞輪詢 KV sender）
5. KV 經 RDMA 推送（TransferQueue；後端 Mooncake 或 NIXL）
6. decode 拿到 KV，**跳過 prefill forward** 直接組成 decode batch（PrebuiltExtendBatch → RunningBatch）
7. 串流從 decode 回傳；逾時預設 300 s

〔可查證：`python/sglang/srt/disaggregation/{prefill,decode}.py`、`base/conn.py`〕

**vLLM + NIXL——串行送、先算完、再拉**：

1. proxy（或 llm-d sidecar）先把請求送給 prefill，prefill 只產第一個 token
2. prefill 把 KV 留在自己 GPU 上，回傳 KV 的位址（block ids、engine id、host／port）
3. proxy 把位址附在請求上轉給 decode
5. 第一次接觸時經 ZMQ side channel 交換 NIXL metadata（lazy handshake）
6. **decode 用單邊 RDMA read 從 prefill 的 GPU 記憶體「拉」KV**
7. decode 串流回傳；prefill 在 KV 被讀走（或逾時）後才釋放 block

〔可查證：vLLM NIXL connector 文件〕

**講法**：「同一件事兩種寫法——這就是『KV 所有權轉移』：一包 KV 從 prefill 的 HBM 換到 decode 的 HBM。SGLang 讓 decode 先把位子佔好再開算；vLLM 讓 prefill 先算完、把 KV 留著等人來拉。」

### 第 19 頁 · KV 交接要多久（V2 長條）

以 DeepSeek 官方統計的**平均 KV 長度 4,989 token** 計算：

| 注意力 | 每 token KV | 一個請求 | 跨機 400G（49.5 GB/s） | NVL72 域內 NVLink 5（每方向 900 GB/s） |
|---|---|---|---|---|
| MHA（同骨架假想：128 頭 × 128 維） | ~4.0 MB | ~20 GB | **~400 ms** | ~22 ms |
| GQA（Llama-3-70B：8 KV heads × 80 層） | 320 KB | 1.6 GB | ~33 ms | ~1.8 ms |
| **MLA BF16（DeepSeek-V3）** | **70 KB** | **351 MB** | **~7 ms** | ~0.4 ms |
| MLA FP8 | 35 KB | 175 MB | ~3.5 ms | ~0.2 ms |

〔推算：MLA ＝ 61 層 ×（kv_lora_rank 512 ＋ rope 64）＝ 35,136 值／token；跨機頻寬取 llm-d 2026-06-23 400G IB 實測（UCX）；未計握手、排程與協定開銷〕

對照：SGLang 96×H100 的 TTFT 是 **2–5 秒**（含排隊）→ **7 ms 的交接在 TTFT 裡是雜訊**。主角的 H100 叢集 prefill 與 decode 在不同台機器，只能走跨機那一欄。

> **講法**：「前半段說 MLA 是為了 decode 的 HBM 頻寬發明的。它還順手付了另一張帳：**讓 prefill 和 decode 可以放在不同機櫃**。換成 MHA，光交接就要 0.4 秒。」

### 第 20 頁 · 第四站：decode 一步（V3 一層的資料流）

> 🎛 **教具對應**：第 3 層「Decode 一步」，可拉 EP 大小、切 dispatch 精度，看一個 token 的 8 個專家落在哪幾張卡。

一層 MoE layer 在 DP-attention + EP 下：

1. **Attention 資料平行**：每張卡只算自己那批請求的 MLA（FlashMLA kernel），KV 只存在自己卡上 → **不重複**（TP 下 MLA 的 latent 無法按 head 切，每張卡都得各存一份）
2. **Gate**：每個 token 選 8 個 routed expert
3. **Dispatch**（DeepEP low-latency，FP8）：token 送去專家所在的卡——**同一張目的卡只送一次**，所以平均送往約 7.5 張卡
4. **Experts**：各卡用 grouped GEMM 算自己的專家（DeepGEMM）
5. **Combine**（BF16）：結果送回原卡
6. × 58 層 → LM head → 取樣 → 串流回使用者

- **DeepEP low-latency 模式**：純 RDMA（NVSHMEM + IBGDA，GPU 直接敲網卡門鈴、不經 CPU）、固定預配置 buffer → 可被 CUDA Graph 錄下來（接第三堂④）。〔可查證：DeepEP README〕
- decode 每張 GPU 只放 **4 個 routed 專家**（288 槽 ÷ 72 卡）+ 1 個 shared。〔推算自 LMSYS 的 288 槽〕
- 前 3 層 dense FFN 也走資料平行。〔可查證：LMSYS〕

### 第 21 頁 · 通訊的時間帳（V2 堆疊長條）

DeepEP low-latency 實測（H800 + CX7 400G、每批 128 token、hidden 7168、top-8、FP8 dispatch）：

| EP 大小 | dispatch | combine | ×58 層＝每步通訊 |
|---|---|---|---|
| 8 | 77 µs | 114 µs | ~11 ms |
| 32 | 155 µs | 273 µs | ~25 ms |
| 128 | 192 µs | 369 µs | ~33 ms |
| 256 | 194 µs | 360 µs | ~32 ms |

〔單層數字可查證：DeepEP v1.2.1 README；×58 為推算〕

**主角的帳**（EP72、每卡 256 序列，套教具模型，見 §6）：

| 項目 | 數字 | 強度 |
|---|---|---|
| 實測一步（TBO 開） | 8 × 256 ÷ 22,282 ≈ **92 ms**（原文 ITL ≈ 100 ms） | 單一來源 |
| 模型推算：每層通訊 | ≈ 0.86 ms → × 58 ≈ **50 ms** | 推算 |
| 反推：不開 TBO 的一步 | 92 × 1.35 ≈ **124 ms** → 通訊佔 **約四成** | 推算 |
| 反推：計算本身 | 124 − 50 ≈ **75 ms** | 推算 |

所以引擎在做的三件事，全都是「把等待藏起來」：

| 手法 | 做什麼 | 數字 |
|---|---|---|
| Two-batch overlap（SGLang）／dual-batch overlap（vLLM）／5 段 pipeline（DeepSeek） | 一個 micro-batch 通訊時，另一個在算 | decode 吞吐 **+35%**（LMSYS，128 序列／卡、模擬 MTP 條件） |
| 通訊降精度 | dispatch 用 FP8 → NVFP4 | NVFP4 dispatch 讓 all-to-all 流量 **÷4**（vLLM GB200 blog，相對 BF16）、**減半**（LMSYS GB200 Part II，相對 FP8） |
| 固定 buffer + CUDA Graph | 每步不重新配置、不回 CPU | 接第三堂④ |

> 💡 **一個 2026 年的轉折**：DeepEP V2（2026-04-29）改用 NCCL 後端、支援到 EP2048，並**拿掉了「零 GPU 開銷」的純 RDMA low-latency 模式**。〔可查證：DeepEP repo〕→ 開講前確認你用的框架版本走哪條路。

### 第 22 頁 · 為什麼拆散便宜（回答鉤子的一半）

**單卡 HBM 佔用（80 GB H100）**：

| | TP16（2 台一組） | DP attention + EP72（9 台） |
|---|---|---|
| 每卡權重 | ≈ 43 GB（688.6 GB ÷ 16） | ≈ 29 GB：4 routed + 1 shared 專家 × 58 層 ≈ 12.8 GB（FP8）；attention 等非專家權重整份複製 ≈ 12.6 GB（FP8）；embedding + LM head ≈ 3.7 GB（BF16） |
| 每卡剩給 KV | ≈ 37 GB，但 MLA latent 無法按 head 切，**16 張卡各存同一份** | ≈ 51 GB，**只存自己那批請求** |
| 72 張卡能放的「不重複」KV | 4.5 組 × 37 GB ≈ **170 GB** | 72 × 51 GB ≈ **3.6 TB** |

〔推算；未扣 activation、CUDA Graph 與 DeepEP buffer，只看量級〕

- **省下的 HBM 拿去放 KV → batch 開大**：decode 每張 H100 跑 **256 條序列**。〔單一來源；交叉驗算：22,282 tok/s ÷ 8 卡 ÷ 256 ≈ 每步 92 ms，與原文 ITL ≈ 100 ms 吻合〕
- **接第三堂的尺**：decode 是 memory-bound，AI ≈ B；H100 的 ridge point ≈ 296，**B 要拉到幾百**才吃得滿算力——大規模 EP 是把 B 推到幾百的方法。
- **EPLB**：熱門專家做副本（32 個冗餘槽）→ decode **2.54×**、prefill **1.49×**。〔單一來源：LMSYS〕
- 實測結果：decode **vs TP16 為 5.2×**。〔單一來源：LMSYS〕

### 第 23 頁 · 代價：節拍器

- DP-attention + EP 下，**同一個 decode 單元的所有 rank 必須同步進入每一層的 all-to-all**——沒請求的卡也得跑空批次陪跑（集合通訊的本質）。
- → **最慢那張卡決定整體 ITL**（分散式系統的 barrier／straggler 問題）
- → **掉一張卡，整個 72 卡單元停擺**（爆炸半徑隨 EP 變大，容錯只能以整個單元為單位）
- → 專家負載傾斜時，熱門專家那張卡就是 straggler → EPLB 定期重排，**這是權重唯一會「搬家」的時候**：TRT-LLM 實測單一 FP4 專家 24 MiB，最多重排 **348 GiB** MoE 權重。〔單一來源：TRT-LLM blog〕

> **講法**：「拆散換來了大 batch，代價是 72 張卡變成一支軍樂隊——所有人踩同一個節拍。節拍能踩多快，取決於傳令兵跑多快。接下來我們換幾種硬體，看傳令兵的速度差多少。」

### 第 24 頁 · 互動環節指引頁

切出去開 [../interactive/rack-journey-map.html](../interactive/rack-journey-map.html)，建議動線（約 6 分鐘）：

| 步 | 教具操作 | 要讓聽眾看到的現象 |
|---|---|---|
| 1 | 第 1 層全景，停在 H100，點「③ MoE 交換」 | decode 池的交換九成要出機器、走 RDMA |
| 2 | 按 `G` 切 GB200 NVL72 | 同一種資料，整段路縮回機櫃內 |
| 3 | 第 2 層 KV 交接，attention 切 MHA → MLA | 400 ms → 7 ms；在 TTFT 的 2–5 s 裡變成雜訊 |
| 4 | 第 3 層 Decode 一步，EP 從 8 拉到 144（H100） | EP8 權重放不下；EP 越大、跨機比例越高、每卡權重越少 |
| 5 | 第 4 層時間帳，依序看 H100 → B200 → GB200 | **NVLink 快一倍（H100 → B200），通訊一毫秒都沒省；域從 8 變 72（B200 → GB200），通訊 50 → 9 ms** |

> **講法**：「三種硬體跑同一個請求，router、prefill、KV 交接幾乎不動——只有 decode 的交換那一站變了。而且讓它變的不是頻寬，是『一個域裡有幾張卡』。下一頁解釋為什麼。」

### 第 25 頁 · 合回來：真瓶頸

回到第 15 頁的四種資料：

- 請求本身 → 前端乙太就夠 ✅
- KV 交接 → MLA 讓它 7 ms，跨機櫃也行 ✅
- 權重 → 幾乎不搬 ✅
- **MoE 交換 → 每字 116 次、每層都要全員同步 ❌ 只有它付不起慢車道**

量化「慢車道佔多少」（均勻路由假設）：

| 部署 | 一個 token 送往的卡裡，落在同一個 scale-up 域的比例 | 走跨機 RDMA 的流量 | 每步通訊（教具模型） |
|---|---|---|---|
| H100 8 卡機 × 9 台，EP72（主角） | (8 − 1) ÷ (72 − 1) ≈ 10% | **≈ 90%** | ≈ 50 ms |
| B200 8 卡機 × 9 台，EP72 | 同上 | **≈ 90%** | ≈ 50 ms（NVLink 快一倍也沒用） |
| GB200 NVL72，EP72 | 100% | **0%** | ≈ 9 ms |

- 頻寬差：NVLink 5 每方向 **900 GB/s**，是 400G 網卡（50 GB/s）的 **18 倍**。（NVIDIA 行銷寫「36×」，是拿雙向合計比單向。）〔可查證：NVIDIA 2025-06-06 blog〕

〔推算：DeepSeek-V3 有 group-limited routing（8 組選 4 組），實際跨機比例依部署拓樸與 EPLB 擺法而定；此處只為建立量級直覺〕

> **本堂最重要的一句**：MoE 推論的硬體選型，第一個要看的數字不是 FLOPs、也不是單卡 HBM，而是**一個 scale-up 域裝得下幾張卡**。

### 第 26 頁 · 證據（V2）

SemiAnalysis InferenceX（DeepSeek-R1 FP4、1K/1K、Dynamo + TRT-LLM、MTP、**125 tok/s/user**）：

| | 同一顆 GPU？ | 每卡 NVLink | NVLink 域 | 拓樸 | tok/s／GPU | $／1M token |
|---|---|---|---|---|---|---|
| B200（HGX 8 卡） | Blackwell | 1.8 TB/s（雙向） | **8** | 4 prefill + 40 decode（EP8） | 941 | $0.576 |
| GB200 NVL72 | Blackwell | 1.8 TB/s（雙向） | **72** | 8 prefill（TP8）+ 16 decode（EP16） | **4,130** | **$0.149** |

〔可查證：inferencex.semianalysis.com 2026-05-23〕

→ **同一顆晶片、同樣的每卡頻寬**，差別幾乎只剩域的大小（另有 Grace CPU vs x86 的差異，影響較小）：**每卡 4.4×、成本 ÷3.9**。
→ 教具模型只算出通訊那一段的差（一步約 1.2×（TBO 開）～1.5×（TBO 關））；其餘來自域變大後 EP 能開更大、每卡權重更少、batch 更大（第 22 頁的機制）。

旁證：LMSYS GB200 NVL72 Part II，FP8 attention + NVFP4 MoE：**26,156 prefill / 13,386 decode tok/s／GPU**，相對 H100 為 3.8× / 4.8×（這組同時換了 GPU 世代，只當旁證）。
### 第 27 頁 · 這趟旅程的每一站，都能在前面找到它的那一頁（V1）

| 旅程的一站 | 決定它快慢的東西 | 在哪裡 |
|---|---|---|
| router 選誰 | cache-aware、局部性 vs 均衡 | 本堂第 16 頁 |
| prefill 跨機 | compute-bound、大 GEMM | 第一堂 roofline |
| KV 交接 7 ms | MLA 把 KV 壓到 70 KB | 本堂旋鈕①（第 4 頁） |
| KV 放得進 HBM | 分頁 KV、continuous batching | 第三堂② |
| decode 為什麼要大 batch | memory-bound、AI ≈ B | 第一堂、第三堂地基 |
| 116 次 all-to-all 走哪條線 | scale-up 域 vs RDMA 頻寬階梯 | 第二堂 Part B |
| 每步不回 CPU | CUDA Graph、固定 buffer | 第三堂④ |
| 專家熱點、掉卡 | EPLB、爆炸半徑 | 本堂第 22–23 頁 |
| 每字少搬幾個 byte | MoE 稀疏、FP8 / NVFP4 | 本堂旋鈕②⑤（第 6、9 頁） |

> 「第一堂那張記憶體階層表，在最後一堂長成了一座資料中心。」

### 第 28 頁 · 帶走三句話（閉環）

1. **架構決策就是硬體帳單。** MLA 為了 decode 的 HBM 頻寬而生，順手讓 KV 交接只要 7 ms；MoE 讓每個字少算，卻換來每字 116 次的全員交換。模型怎麼設計，決定了機櫃上要搬什麼。
2. **72 張卡同步踏一步划算，是因為最常搬的東西走了最快的路。** 每字 116 次的 MoE 交換被 DeepEP 壓到每層不到 1 ms；一次性的 KV 交接小到可以跨機櫃。
3. **拆散的收益在 HBM，代價在節拍——所以選硬體先看域的大小。** 每卡只放 4 個專家、跑 256 條序列，decode 比 TP16 快 5.2×；但付不起慢車道的交換只能靠更大的 scale-up 域：同一顆 Blackwell，域從 8 到 72，每卡 4.4×。

> **全系列收束**：第一堂硬體的尺 → 第二堂單卡到多卡 → 第三堂單機引擎 → 第四堂從模型架構到一整排機櫃。**同一個敵人，四個高度。**

---

## 4. 視覺決策（閘門：V1–V4）

| 頁 | 圖 | 通過條件 | 刪掉要多講幾句？ |
|---|---|---|---|
| 2 | prefill → decode 自回歸流程 | V3（有迴圈、兩階段性質不同） | ≥4 句 |
| 5 | 256 格專家點亮 8 格 | V2（稀疏比例一眼可見） | ≥3 句 |
| 13 | 每卡吞吐長條 | V2（三組以上數字比較） | ≥3 句 |
| 14 | 機櫃拓樸 | V1（池子、網路、機櫃的位置關係） | ≥5 句方位詞 |
| 15 | 四種資料 × 四條路 | V2（量級差 + 頻率差） | ≥4 句 |
| 18 | 三泳道時序圖 | V3（>4 步、有並行分支、兩種寫法） | ≥6 句 |
| 19 | KV 傳輸時間長條 | V2（跨兩個數量級） | ≥3 句 |
| 20 | 一層資料流 | V3（6 步、含跨卡回流） | ≥5 句 |
| 21 | 一步時間堆疊 | V2 | ≥3 句 |
| 22 | 單卡 HBM 佔用 | V2 | ≥3 句 |
| 24 | 教具截圖（第 4 層時間帳） | V2 | 指引頁本身 |
| 25 | 跨機流量比例 | V1 + V2 | ≥4 句 |
| 26 | B200 vs GB200 | V2 | ≥2 句 |
| 27 | 旅程路線 × 堂次 | V1 | ≥5 句 |

不放：機櫃照片（除非講者有現場照並圈出 NVLink spine／compute tray）、廠商 logo、裝飾圖。

---

## 5. 待補來源 / 開講前要再對一次的

- InferenceX 儀表板數字持續更新，**開講前對一次**。
- DeepEP V2 拿掉純 RDMA low-latency 模式後，SGLang / vLLM 的 decode 預設走哪個後端——確認版本。
- 公開文獻只有 KV 傳輸**頻寬**，沒有端到端交接**毫秒數**；第 19 頁的 ms 全是推算。
- LMSYS 96×H100 原文**未載明網卡配置**；本堂以 DGX H100 典型的每卡一張 400G 估。
- TBO 的 decode +35% 是在「128 序列／卡、模擬 MTP」條件下量的；第 21 頁用它反推 124 ms 是跨條件套用。
- Vera Rubin 的命名在 CES 2026 前從 NVL144 改回 **VR NVL72**（72 個封裝）。第二堂講稿與投影片已同步（2026-09-21：「NVL144/CPX 版」改寫為「Rubin CPX」）。
- 前半段各家模型數字（第 6、10 頁）：2026 上半年版本迭代極快，開講前對一次官方頁面。

---

## 6. 資料來源

- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)（608B/342B/168B、278 node、EP32/EP144、平均 KV 長度 4,989）
- [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)（61 層、前 3 層 dense、256+1 專家選 8、MLA 512+64）
- [DeepEP v1.2.1 README](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/README.md)（low-latency dispatch/combine µs、normal 模式頻寬）｜[DeepEP V2](https://github.com/deepseek-ai/DeepEP)
- [LMSYS：96×H100 大規模 EP + PD 分離](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)（本堂主角）｜[GB200 Part I](https://www.lmsys.org/blog/2025-06-16-gb200-part-1/)｜[GB200 Part II](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/)｜[GB300 長上下文](https://www.lmsys.org/blog/2026-02-19-gb300-longctx/)｜[DeepSeek-V4 day-0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang PD 分離文件](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)｜[sglang_router mini_lb.py](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py)
- [vLLM：大規模服務（H200 2.2k tok/s/GPU）](https://vllm.ai/blog/2025-12-17-large-scale-serving)｜[vLLM：DeepSeek-R1 on GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)｜[vLLM：DeepSeek-V4](https://vllm.ai/blog/2026-04-24-deepseek-v4)｜[NIXL connector](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [llm-d wide EP](https://llm-d.ai/docs/well-lit-paths/foundations/wide-expert-parallelism)｜[llm-d 網路實測](https://llm-d.ai/blog/networking-for-distributed-inference-llm-d)
- [NVIDIA Dynamo 1.0](https://nvidianews.nvidia.com/news/dynamo-1-0)｜[TRT-LLM 大規模 EP](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html)
- [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)｜[GB200 NVL72 × Dynamo（36× 說法）](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models)｜[NVL72 AI factory 參考架構（GB300）](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html)｜[Vera Rubin 平台](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)｜[Vera Rubin 量產宣布](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory)｜[DGX H100](https://www.nvidia.com/en-eu/data-center/dgx-h100/)｜[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [InferenceX：GB200 NVL72 vs B200](https://inferencex.semianalysis.com/blog/gb200-nvl72-vs-b200-disagg-deepseek-r1-fp4-dynamo-trt)
- [DeepSeek-V3 技術報告](https://arxiv.org/abs/2412.19437)（MLA、DeepSeekMoE、無 aux loss 均衡、MTP、FP8、DualPipe）
- DeepSeek-V2（MLA −93.3%）｜DeepSeek-V3.2-Exp（DSA，2025-09，API 降價 >50%）｜[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（CSA/HCA/mHC、1M ctx 下 FLOPs 27%、KV 10%）
- [Kimi K2 技術報告](https://arxiv.org/abs/2507.20534)（1T-A32B、MuonClip）｜[Kimi K3 模型卡](https://huggingface.co/moonshotai/Kimi-K3)（2.8T-A104B、69 KDA + 24 Gated MLA、896/16、MXFP4/MXFP8 QAT、1M ctx）
- [LMSYS：No Free Lunch — Deconstruct Efficient Attention with MiniMax M2](https://www.lmsys.org/blog/2025-11-04-miminmax-m2/)｜[MiniMax 官方說明](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)｜[MiniMax-M1 論文](https://arxiv.org/abs/2506.13585)
- Qwen3 / Qwen3-Next / Qwen 3.5、GLM-5 —— 各家官方部落格；綜覽見 [Interconnects 開源模型整理](https://www.interconnects.ai/p/latest-open-artifacts-19-qwen-35)
- DeepSeek 開源週零件：FlashMLA / DeepEP / DeepGEMM / EPLB / 3FS（[open-infra-index](https://github.com/deepseek-ai/open-infra-index)）

---

## 7. 互動教具的計算模型（講者備查）

教具 [rack-journey-map.html](../interactive/rack-journey-map.html) 的數字都由下面幾條算式產生；被問「這數字哪來的」時照這裡回答。

**KV 交接時間**＝每 token KV × token 數 ÷ 鏈路頻寬（未計握手與協定開銷）。

**MoE 每層通訊**（decode、每卡 B = 256 序列）：

1. 一個 token 送往的不同卡數 u ＝ EP ×（1 −（1 − 1/EP）⁸）；扣掉自己，遠端約 u ×（EP − 1）/ EP 張（EP72 時 ≈ 7.5）
2. 其中落在同一個 scale-up 域的比例 ＝（域大小 − 1）/（EP − 1）
3. 每段（dispatch、combine）時間 ＝ max（域內位元組 ÷ scale-up 每方向頻寬，跨域位元組 ÷ 網卡頻寬）
4. 每層通訊 ＝ dispatch ＋ combine ＋ 固定開銷 0.11 ms；每步 ＝ × 58 層

**校準**：拿 DeepEP README 的實測條件（128 token、FP8、純 RDMA 50 GB/s）代入第 1–3 步：

| EP | 模型純線速 dispatch / combine | DeepEP 實測 dispatch / combine |
|---|---|---|
| 32 | 128 / 255 µs | 155 / 273 µs |
| 128 | 142 / 283 µs | 192 / 369 µs |
| 256 | 144 / 288 µs | 194 / 360 µs |

→ 純線速解釋了實測延遲的 75–95%，殘差取整成每層 0.11 ms 固定開銷。

**一步的時間帳**：計算時間固定為 74.5 ms、TBO 能藏住 65% 的通訊——兩個數由主角實測反推（一步 92 ms、TBO +35%、模型通訊 50 ms）。**換硬體時刻意不改計算時間**，只改互連，用來隔離「域大小」這個變數。

**每卡權重**＝（288 ÷ EP ＋ 1）× 單專家槽 2.55 GB（44M 參數 × 58 層，FP8）＋ 非專家權重 16.3 GB。

| 硬體 | 域大小 | scale-up 每方向 | 網卡 | HBM |
|---|---|---|---|---|
| H100 8 卡機 | 8 | 450 GB/s（NVLink 4） | 400G ≈ 50 GB/s | 80 GB |
| B200 8 卡機 | 8 | 900 GB/s（NVLink 5） | 400G | 180 GB |
| GB200 NVL72 | 72 | 900 GB/s（NVLink 5） | 400G | ~186 GB |

---

## Q&A 速查

**Q：MoE 在單卡上不是不省記憶體嗎？稀疏化到底省什麼？**
A：省每個字的權重讀取與 FLOPs，不省容量；容量要靠大規模 EP 把專家攤到很多卡上（第 22 頁）。

**Q：長 context 那麼貴，稀疏注意力是不是必然的未來？**
A：方向上是，但注意 MiniMax 的教訓：**線性**注意力目前在多跳推理與生態相容性上仍有實證問題；**稀疏**比較被接受，因為它不改變「KV 還在」這個前提。2026 的實務共識是**混合 + 稀疏**。

**Q：這些模型我在本機跑得動嗎？**
A：總參數是容量門檻、活躍參數是速度門檻——兩者都要看。K3 的 2.8T 即使 MXFP4 也要 ~1.4 TB 才裝得下權重，本機無望；但 Qwen3-Next 80B-A3B 這種「大總參數、小活躍」的設計，在統一記憶體機器（Apple M 系列大容量）上是可行的——這正好呼應第一堂的「容量夠、頻寬低 → 慢但跑得動」。

**Q：為什麼沒講 AMD？**
A：2026-09-22 決定把 NVIDIA vs AMD 的選型拿掉，讓後半段專心講「域的大小」這一個變數。要比較時的判斷式是：單位是一台看單台總 HBM，單位是機櫃看 scale-up 域大小 × 軟體成熟度。

---

## 附錄 A：未上投影片的延伸素材——1M context

> 2026-09-14 決定不上投影片，保留在這裡供 Q&A。

- DeepSeek-V4-Pro **1.6T-A49B**、V4-Flash **284B-A13B**，都支援 1M context。〔可查證：NVIDIA blog 2026-04-24〕
- vLLM day-0：1M context 下 **每條序列 KV 9.62 GiB**（BF16），V3.2 式架構則要 **83.9 GiB**。〔可查證：vLLM blog 2026-04-24〕
- 換算跨機 400G（49.5 GB/s）：V4 **~0.2 s**、V3.2 式 **~1.8 s**。〔推算〕→ 第 15 頁 KV 那一列，在 1M context 下從「搬得起」退回「要算一算」。
- SGLang day-0 對 V4 的回應正在這一列：**ShadowRadix** 前綴快取、**HiSparse**（KV 卸載到 CPU）、**context parallel** attention。〔可查證：LMSYS 2026-04-25〕
- 硬體端：NVIDIA CMX（第二堂）把 KV 卸載到 BlueField-4 + flash。
