# 第二堂課 講稿／索引 — Transformer × GPU：逐 block 上機（單卡）→ 多卡的資料平行與互連

投影片：[../slides/class2-transformer-gpu.html](../slides/class2-transformer-gpu.html)（24 頁）｜重建：`cd ../slides/build && node generate_class2.js`
互動地圖 ×3：
- [gpu-map.html](../interactive/gpu-map.html)（第 4 頁指引）：Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 下鑽。
- [transformer-map.html](../interactive/transformer-map.html)（第 11 頁指引）：玩具 **decoder-only** Transformer，decoder 全景 → Block（masked self-attn + FFN）→ Attention → Head → 計算子(L2⟷HBM) → FlashAttention(線上 softmax) → 硬體。
- [parallelism-map.html](../interactive/parallelism-map.html)（第 23 頁指引，本堂新做）：同一個玩具 Transformer 攤到多卡，單卡 → 裝不下 → DP → TP → PP → 互連硬體；TP/PP 兩層把「一層」的權重矩陣（Wq/Wk/Wv/Wo + W1/W2）怎麼被切畫出來。

> 本堂**聚焦「Transformer × GPU 框架」**，兩條線：**Part A** 把玩具 Transformer 逐 block 對應到 NVIDIA GPU 單元（單卡跑得完）；**Part B** 模型大到一張卡裝不下時，資料平行的問題與 NVIDIA 的跨 GPU 通訊新技術。延續第一堂（[full-series.md](full-series.md)）的 roofline / 記憶體階層 / KV cache / decode memory-bound。

---

## 頁面地圖（24 頁）

| 頁 | 內容 | 對應 |
|---|---|---|
| 1 | 標題：從一張卡的每個 block，到多卡的通訊 | — |
| 2 | 路線圖：Part A 單卡逐 block × Part B 多卡通訊 | — |
| **Part A · 單卡逐 block × GPU** | | |
| 3 | 回顧玩具 Transformer（T=6、d=6、2 heads，同構真實模型） | transformer-map |
| 4 | 我們要對應的那台 GPU：運算單元 × 記憶體階層 | 🔍 互動① gpu-map |
| 5 | Block 0 Embedding + 位置編碼：查表 gather → **CUDA core / memory-bound** | |
| 6 | Block 1 QKV 投影：X·W = GEMM → **Tensor Core / compute-bound** | |
| 7 | Block 2 多頭切分：head 軸＝平行軸 → batched GEMM / warp | |
| 8 | Block 3 Attention：Q·Kᵀ→softmax→·V；**prefill=compute / decode=memory**，KV cache 住 HBM | |
| 9 | Block 4 Add & LayerNorm：逐元素 + 歸約 → **CUDA core / memory-bound**、kernel fusion | |
| 10 | Block 5 FFN 6→12→6：兩個大 GEMM → **Tensor Core / compute-bound**、token 軸平行 | |
| 11 | 彙整表：block → 運算 → GPU 單元 → 判定 | 🔬 互動② transformer-map |
| 12 | Part A 收束：權重 + KV + activation 都在一顆 HBM → 一張卡跑得完 | |
| **Part B · 多卡：資料平行與 NVIDIA 互連** | | |
| 13 | 為什麼一張卡不夠：權重（70B≈140GB）+ KV cache + 訓練狀態（參數 3–4×） | |
| 14 | 資料平行 DP：複製模型、切 batch、梯度 all-reduce | |
| 15 | **DP 的四個問題**（本堂重點）：①沒變小 ②通訊隨規模長 ③狀態重複 ④decode 延遲無解 | |
| 16 | 切模型本身：TP（層內）/ PP（層間）/ EP（MoE）taxonomy | |
| 17 | 通訊變瓶頸：頻寬階梯 × 平行策略落位 | |
| 18 | scale-up：NVLink 5 / NVSwitch / GB200 NVL72 + **Rubin/NVLink 6（2026）** | |
| 19 | in-network：NVSwitch + **SHARP**（把 all-reduce 搬進交換器）+ NCCL | |
| 20 | scale-out：InfiniBand / Spectrum-X + GPUDirect RDMA | |
| 21 | 組起來：一個大模型怎麼跑在 NVL72 上（DP×TP×PP×EP 混合） | |
| 22 | **CMX（2026）**：把 KV cache 卸到 context tier，呼應 decode memory-bound | |
| 23 | 互動環節③：多卡平行地圖 | 🕸️ 互動③ parallelism-map |
| 24 | 帶走三句話 | |

---

## Part A 速查：逐 block → GPU 單元 → 被誰卡住

玩具規格 T=6、d_model=6、heads=2、d_head=3、FFN hidden=12（同構於 GPT 級的 d=12288、96 heads、128K）。

| # | Block | 主要運算 | GPU 單元 | 資料/搬運 | 判定 |
|---|---|---|---|---|---|
| 0 | Embedding + 位置 | 查表 gather + 逐元素加 | CUDA core | embedding 表在 HBM、只讀幾列 | **memory-bound**（運算極少） |
| 1 | QKV 投影 | GEMM：X·Wq/Wk/Wv | **Tensor Core**（MMA） | tile 留 shared 重複用 | **compute-bound**（T 夠大） |
| 2 | 多頭切分 | 拆平行軸（6→2×3） | warp / batched GEMM | — | 結構（多一個平行軸） |
| 3 | Attention | Q·Kᵀ→softmax→·V | Tensor（矩陣乘）+ CUDA（softmax） | KV cache 住 HBM | **prefill=compute / decode=memory** |
| 4 | Add & LayerNorm | 逐元素加 + 歸約 | CUDA core | 整份 activation 進出 HBM | **memory-bound**（算少搬多） |
| 5 | FFN | 兩個 GEMM（6→12→6） | **Tensor Core** | tile 留 shared | **compute-bound** |

**一句話**：一層裡「大矩陣乘（QKV、FFN）吃算力、其餘（embedding、softmax、norm）吃頻寬」。這節奏乘上倍率就是真實模型。decode 時連 attention 的矩陣乘都退化成 GEMV → 整層 memory-bound（第一堂「<5% 利用率之謎」）。

---

## Part B 速查：四種平行、切什麼、走哪條線

| 平行 | 切什麼 | 每卡模型 | 通訊型態 | 頻寬需求 | 落位 |
|---|---|---|---|---|---|
| **DP** 資料平行 | 資料（batch） | 整份（沒變小） | 梯度 all-reduce / 每步 | 中（≈2×參數量/步） | 可跨節點（scale-out） |
| **TP** 張量平行 | 層內（矩陣切片） | 變小 1/N | **每層 all-reduce ×2** | **極高** | **必須 NVLink 域（scale-up）** |
| **PP** 管線平行 | 層間（stage） | 變小 1/N | p2p 傳 activation | 中（僅 stage 邊界） | 可跨節點；代價是 bubble |
| **EP** 專家平行 | MoE 專家 | 專家散開 | all-to-all 路由 | 高 | scale-up 為主 |

**資料平行的四個問題（本堂重點，第 15 頁）**：
1. **模型沒變小**：每張卡仍放整份模型 → 「一張卡裝不下」的模型 DP 根本救不了。
2. **通訊隨規模長大**：梯度 all-reduce 每步搬 ≈ 2× 參數量；模型越大、卡越多，同步越兇。
3. **記憶體重複浪費**：權重/梯度/optimizer 狀態每卡各存一份 → ZeRO/FSDP 把它們切開（分片），但換來更多通訊。
4. **推論延遲無解**：decode 是 memory-bound，DP 只增吞吐、對單一請求延遲沒幫助，KV cache 還各卡獨立。

→ 所以要切「模型本身」（TP/PP/EP）。實務是 **3D/4D 混合**（DP×TP×PP(×EP)）：切得越細、卡間通訊越密 → 通訊成了新瓶頸。

---

## NVIDIA 互連速查（Blackwell 為主 + Rubin/CMX 為 2026 最新）

數字為約略值，以官方規格為準。scale-up＝機架內用「記憶體語意」把多顆綁成一顆大 GPU；scale-out＝跨節點用「訊息語意」（RDMA）。

### scale-up（NVLink / NVSwitch）— TP / EP 的家
- **NVLink 4（Hopper, H100/H200）**：~900 GB/s / GPU（**雙向合計**；每方向 450 GB/s）。
- **NVLink 5（Blackwell, GB200）**：**1.8 TB/s / GPU**（雙向合計；18 條 × 100 GB/s，≈ PCIe Gen5 雙向的 14×；每方向 900 GB/s）。
- ⚠️ **單位約定**：NVIDIA 規格頁的 NVLink 數字都是**雙向合計**；IB / 乙太的 400G ≈ 50 GB/s 是**每方向**。本堂投影片照官方寫法（已標「雙向」），[第四堂](class4-models-to-racks.md)計算單向傳輸時間時一律換成每方向（NVLink 4 = 450、NVLink 5 = 900 GB/s）。
- **NVSwitch（4 代）**：72 個 NVLink 5 埠 / 晶片；**GB200 NVL72＝72 顆一個 NVLink 域、130 TB/s 聚合**；fabric 可擴到 576 GPU / 1 PB/s。
- **NVLink 6（Rubin, 2026 最新）**：**3.6 TB/s / GPU**（雙向；NVLink 5 的 2×）；**Vera Rubin NVL72＝260 TB/s 聚合**（CES 2026 前由舊名 NVL144 改回 NVL72，數的是 72 個封裝）。CES 2026（1/5）發表、GTC 2026（3/16）、H2 2026 出貨；第四堂引用 2026-05-31 宣布量產、秋季出貨。**Rubin CPX** 把 **prefill 拆出來**專做（即 PD 分離，第四堂問題⑥）。

### in-network（SHARP + NCCL）
- **SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）**：把 all-reduce / reduce / broadcast 的加總**直接在 NVSwitch / IB 交換器的 ASIC 裡算完** → 省 NVLink 頻寬、也把 GPU 的 SM 解放出來算模型。
- **NCCL**：集合通訊的軟體層（all-reduce / all-gather / reduce-scatter / all-to-all）；**2.27 起 NVLink 與 IB 都能吃 SHARP**，1000+ GPU 規模訓練受益。DP 的梯度同步、TP 的層內同步都靠 all-reduce → offload 進交換器就鬆一大截。

### scale-out（跨節點網路）— DP / PP 的家
- **InfiniBand（Quantum）**：NDR 400 / XDR 800 Gb/s，原生支援 SHARP。
- **Spectrum-X 乙太**：為 AI 調過的以太網（adaptive routing + 壅塞控制、跑 RoCEv2）；Rubin 世代為 Spectrum-6。
- **GPUDirect RDMA**：遠端 NIC 直接讀寫對方 GPU 的 HBM、繞過 CPU。
- 兩層網路差一個數量級（NVLink ~TB/s ≫ IB/乙太 ~50–100 GB/s/GPU）→「通訊密的平行留機架內、稀的才跨節點」是被頻寬逼的，不是選擇。

### CMX（Context Memory，2026 新招）— 直接打 decode 的頻寬/容量牆
- **BlueField-4 STX** 儲存架構（GTC 2026 發表）：把 KV cache 標準化成三層 **GPU HBM → CPU DRAM → NVMe flash**，由 BlueField-4 DPU 排 I/O，讓 GPU 不必等儲存。
- 用 **Spectrum-X** 的低延遲 RDMA 存取共享 KV cache；**DOCA** 管理、**Dynamo + NIXL** 統籌 prefill/decode/KV，支援 **prefill–decode 拆分**與**前綴重用**。
- 官方數字：**~5× token 吞吐、~4× 能源效率、~2× 資料載入**；H2 2026 出貨。（⚠️ [glossary.md](glossary.md) 舊版寫「~5× 能效」，兩者不一致，開講前以官方頁為準。）呼應第一堂 Part 3（decode memory-bound、KV cache）與 Part 4（prefetch）。

---

## 關鍵推導速查（被追問時用）

- **DP 的 all-reduce 成本**：每步要把每張卡的梯度加總再發回，通訊量 ≈ 2×（N−1)/N × 參數量位元組（ring all-reduce），與 batch 無關、與參數量成正比 → 模型越大同步越貴。這是「② 通訊隨規模長大」的來源。
- **TP 為何必須 NVLink 域**：張量平行在**每一層**的兩個地方要 all-reduce（attention 輸出、FFN 輸出），一次 forward 就數十~上百次；若走 PCIe（~64 GB/s）或跨節點 IB（~百 Gb/s），卡間頻寬直接主宰 step time → 只能待在 NVLink（~TB/s）域內。
- **PP 的 bubble**：管線有 P 個 stage、M 個 micro-batch 時，理想利用率 ≈ M/(M+P−1)；bubble 佔比 ≈ (P−1)/(M+P−1) → 要靠切多一點 micro-batch 把 bubble 壓小。PP 通訊只在 stage 邊界 p2p，所以能容忍較慢的線（可跨節點）。
- **為何 EP「省 FLOPs 不省通訊」**：MoE 每 token 只算少數 expert（省算力），但 token 要 all-to-all 路由到 expert 所在的卡、算完再送回 → 通訊（甚至權重就位）不減反增。又一個「FLOPs ≠ 速度」案例（第一堂 Part 5）。
- **scale-up vs scale-out 一句話**：NVLink 把多顆 GPU 當一顆（記憶體語意、load/store）；網路把多台串起來（訊息語意、RDMA）。頻寬差一個量級 → 張量平行留 scale-up、資料平行才跨 scale-out。

---

## Q&A 速查（現場準備）

- **decode 為什麼多卡也快不了單一請求？** decode 是 memory-bound，瓶頸是「每 token 重讀權重 + KV」÷ 頻寬。DP 增併發（throughput）、TP 能把單層攤到多卡的頻寬上（有幫助但受通訊限），但延遲的物理下限還是搬運量／頻寬——這是 CMX、量化、KV 壓縮（GQA/MLA/DSA）在打的地方。
- **ZeRO / FSDP 算 DP 還是切模型？** 它是「切狀態的 DP」：資料平行的骨架，但把權重/梯度/optimizer 狀態分片到各卡（不再每卡整份），用時再 all-gather → 省容量、換更多通訊。介於 DP 與模型平行之間。
- **一定要 NVLink 嗎？** 通訊稀的平行（DP、PP）在 IB/乙太上可行；通訊密的（TP、EP、頻繁 all-reduce）在純 PCIe 會被卡間頻寬拖死 → 這正是 NVLink/NVSwitch 存在的理由。
- **Rubin 已經能買了嗎？** CES 2026（1/5）發表、GTC 2026（3/16）展開、**H2 2026 才出貨**；現行大量部署的主流仍是 Blackwell（GB200 NVL72 / NVLink 5）。所以投影片以 Blackwell 為主、Rubin 當「最新」收尾。
- **SHARP 到底省什麼？** 省兩樣：把 all-reduce 的加總搬進交換器 ASIC → ①不再佔 GPU 的 SM 去算加法 ②減少 NVLink/IB 上來回搬運的資料量。對 1000+ GPU 的大規模訓練特別有感。
- **CMX 會不會又變成瓶頸？** KV cache 是「短命、可重算」的 AI-native 資料，容忍度高；CMX 用 BlueField-4 排 I/O、Spectrum-X RDMA、NIXL prestage（decode 前把 KV 搬回 HBM），把儲存延遲藏在生成後面（prefetch 的一種）→ GPU 不空等。

---

## 互動地圖使用（第 23 頁指引）

開啟 `interactive/parallelism-map.html`（任何瀏覽器、離線、單一檔案）。
- 操作：數字鍵 `1`–`6` 跳層（單卡 / 裝不下 / 資料平行 / 張量平行 / 管線平行 / 互連硬體）；`Esc` 回上層；URL hash 直達（如 `#tp`、`#pp`、`#net`）。
- 建議講法：`1` 單卡（沒有通訊）→ `2` 模型長大溢出一顆 HBM → `3` DP（點出四個問題）→ `4` TP（先看「一層」= Wq/Wk/Wv/Wo + W1/W2，每個矩陣沿寬切 4 片、每層 all-reduce、標「必須 NVLink 域」）→ `5` PP（同一個「一層」整盒不切、整層分段 + p2p + bubble timeline）→ `6` 互連硬體（scale-up vs scale-out，哪種平行走哪條線、SHARP、CMX；世代只是把數字放大）。切回投影片第 24 頁收尾。

## 檔案與重建

- 投影片：`slides/build/generate_class2.js`（經 pptx-html 轉成 HTML，深色矽晶主題）→ `slides/class2-transformer-gpu.html`（24 頁）。
- 互動地圖：`interactive/parallelism-map.html`（單檔、離線、無相依；6 層，TP/PP 含「一層」權重矩陣解剖）。
- 相關第一堂內容（roofline、KV cache、decode memory-bound、FlashAttention/GQA、TPU/Groq）：[full-series.md](full-series.md)、[transformer-interactive.md](transformer-interactive.md)；縮寫全稱見 [glossary.md](glossary.md)（§11 系統/平行、§12 NVIDIA 互連與 CMX）。

## 來源（互連數字，以官方規格為準）

- NVIDIA GB200 NVL72 / NVLink 5 / NVSwitch：<https://www.nvidia.com/en-us/data-center/gb200-nvl72/>、<https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/>
- Rubin / NVLink 6（CES 2026）：<https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer>
- SHARP in-network computing / NCCL 2.27：<https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/>、<https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/>
- CMX（BlueField-4 Context Memory Storage）：<https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/>、<https://www.nvidia.com/en-us/data-center/ai-storage/cmx/>

## 承上啟下

- **接第一堂**：開場第 2 頁那句「延續第一堂：roofline · 記憶體階層 · KV cache · decode 的 memory-bound」是本堂的前提；Part A 第 8 頁回收第一堂「<5% 利用率之謎」——謎底就發生在 attention 這個 block。
- **留給第三堂的問題**：DP 的第④個問題「推論延遲無解」與「decode 要大 batch 才吃得滿算力」本堂都沒解。Part B 的平行與互連是偏**訓練式**（bulk-synchronous、梯度同步）的視角。
- **下一堂**（[第三堂](class3-engine-single-node.md)）：**先退回一台機器**——推論引擎（SGLang × vLLM）怎麼把 batch 從 1 撐到幾百。多卡、多機的推論問題留到第四堂，並在第四堂 Part A 對照「訓練（本堂）vs 推論服務」的差別。
