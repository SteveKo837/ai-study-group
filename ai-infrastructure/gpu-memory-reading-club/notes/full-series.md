# 合輯索引 — S1–S5 重編成一份投影片的次序與對照

投影片：[../slides/full-series.html](../slides/full-series.html)（34 頁）｜重建：`cd ../slides/build && node generate_full.js`
互動地圖 ×2：[gpu-map.html](../interactive/gpu-map.html)（Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 下鑽；第 4 頁指引）、[transformer-map.html](../interactive/transformer-map.html)（玩具級 **decoder-only** Transformer：7 層由 decoder 全景（自回歸迴圈）→ Block（masked self-attn + FFN）→ Attention → Head → 計算子 matmul(L2⟷HBM 搬運) → **FlashAttention（線上 softmax 6 步驟）** → 硬體，× 三模式 × GPU/TPU/Groq，含 KV cache 串流與 tensor core tiling；第 25 頁指引，報告見 [transformer-interactive.md](transformer-interactive.md)）

合輯不是五場串接，而是**重編去重 + 聚焦**。最新版**聚焦「硬體架構 × Transformer」**：已移除 ASR 案例、NVLink/GPUDirect、Unified Memory 三種，並把「心法」折進記憶體階層頁；另把靜態的「CPU vs GPU」「GPU 解剖」兩頁拿掉，GPU 結構改由互動地圖（已含 SM → CUDA/Tensor core 下鑽）承擔。本檔同時是次序地圖與第一堂的主講稿（原 s1–s5 單場講稿與單場版投影片皆已併入後刪除）。

> **在四堂系列中的位置**：本堂是地基——給出整個系列都會用到的「一把尺」（算術強度 / roofline）與「一張表」（記憶體階層），並用開場謎題「batch=1 decode 算力利用率 <5%」埋下全系列的主敵人：**decode 是 memory-bound**。後三堂都在從不同高度打這個敵人。

## 新次序（五個篇章）

| 頁 | 內容 | 來源 | 整併說明 |
|---|---|---|---|
| 1–3 | 標題、開場謎題、路線圖 | S1-2 + S5-2 | 開場單一謎題（batch=1 decode <5% 利用率） |
| **Part 1 機器** | | | |
| 4 | 🔍 互動環節①（指引頁） | 新增 | 講者切出去開 `interactive/gpu-map.html`，Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 下鑽——**取代靜態的 CPU-vs-GPU / GPU 解剖頁** |
| 5 | warp / SIMT 藏延遲 | S2-4 | |
| 6 | 餵飽一張卡 + Amdahl | S5-3 | 「序列鏈是天花板」整份合輯都要用 |
| 7 | 記憶體階層全景（頻寬）＋「最慢那段路」心法 | S1-8 + S2-5 + S1-9 | 兩張表合一、加「誰管理」欄；原「心法」頁折成一句併入 |
| 8 | 三層記憶體每 GB 價格 | 新增 | DRAM $2–3 / HBM3E $13–17 / 片上 SRAM $5000 起；解釋為何越快越小、KV 為何住 HBM、Groq 為何要很多顆 |
| 9 | 方案比較表 + 選型心法（橫向硬體比較） | S4-8 + S4-9 | 各家加速器（RTX4090/A100/H100/H200/Apple/Grace Hopper）容量·頻寬·定位；接在「每 GB 價格」後湊成「硬體 landscape」一塊（原在 Part 4，移來 Part 1）|
| **Part 2 一把尺** | | | |
| 10–13 | 兩種慢、AI、roofline、ridge point | S1-3,5,6,7 | 原樣保留（S2/S3/S5 的 roofline 回顧頁全刪） |
| 14 | 時脈 × 單元數 × 每 cycle 運算數 | 新增 | 解釋 990 TFLOPS / 3.35 TB/s 怎麼算出來：CUDA vs tensor core 兩條路徑同時脈差 15× |
| 15 | tiling | S2-7 + S2-8 | 兩頁合一（naive/tiled + roofline 右推） |
| **Part 3 模型上機（Transformer 推論）** | | | |
| 16–20 | training、prefill/decode、decode 解謎、KV cache、batch | S3-3,4,5,6,7 | 第 18 頁回收開場謎題；**ASR 案例已移除**（聚焦 Transformer，落地交給互動地圖） |
| **Part 4 資料搬遷** | | | |
| 21 | prefetch / overlap 壓軸 | S4-10 | Part 4 只剩此頁（進出站/PCIe-pinned 頁已移除，與整體脫節）；overlap：把搬運藏在運算後面 |
| **Part 5 共同演化** | | | |
| 22–24 | 平行軸、hardware lottery、Transformer 為平行而生 | S5-4,5,6 | 原樣保留（Amdahl 已前移至第 6 頁） |
| 25 | 🔬 互動環節②（指引頁） | 新增 | 切出去開 transformer-map.html：玩具級 **decoder-only** Transformer（T=5、d=6、2 heads），7 層 decoder 全景 → Block（masked self-attn + FFN）→ Attention → Head → 計算子 matmul(L2⟷HBM) → FlashAttention(線上 softmax) → 硬體，× 訓練/Prefill/Decode × GPU/TPU/Groq |
| 26 | 模型 → 硬體 | S5-7 | |
| 27 | TPU：systolic array | 新增 | weight-stationary、XLA 靜態排程；transformer on TPU；decode 跨硬體一樣 memory-bound |
| 28 | Groq LPU：全 SRAM 砍掉 HBM | 新增 | 回答「KV 塞不下 L2 → 一直從 HBM 串流」：GPU/TPU 痛點（HBM→SRAM 串流）vs Groq 全 SRAM（80 TB/s、多晶片切片） |
| 29–32 | 三個 case（CNN／Transformer／混合）、三問 | S5-8..11 | |
| **收束** | | | |
| 33 | Demo 總表 | 各場 demo 頁 | 五頁 demo 預告 → 一頁（含 M2 實測數字） |
| 34 | 全系列帶走三句話 | S4-12 + S5-13 | 兩場總結合成：尺／搬運／共同演化 |

## 刪掉的重複內容

- 各場「回顧上一場」「下一場預告」頁（S2-2、S3-2、S4-2 的 recap 部分、所有結尾預告卡）
- S3 / S5 的 mini-roofline 回顧（合輯內用「Part 2 的尺」一句話引用）
- S5 對 decode/KV cache 的重述（直接引用 Part 3）

## 聚焦版移除的內容（最新版）

為了把主軸收斂在「硬體架構 × Transformer」，相對於早期 42 頁版另外移除（最終 34 頁）：

- **ASR 案例（Whisper vs CTC + 對照表，2 頁）**：抽象的「decode memory-bound / 平行 vs 序列」改由互動 Transformer 地圖（三模式）落地，比 ASR 更直接。
- **NVLink/GPUDirect、Unified Memory 三種（2 頁）**：偏基礎設施專家向、離主線最遠。
- **進出站：PCIe + pinned/pageable（1 頁）**：host↔device 傳輸機制，與整體（GPU × Transformer）脫節；移除後 Part 4 只剩 prefetch / overlap 壓軸一頁。
- **方案比較表移到 Part 1（第 9 頁）**：它是「超越 H100、各家加速器的橫向比較」，本質是硬體 landscape，與第 8 頁「每 GB 價格」相鄰最順——所以從 Part 4 移到 Part 1 機器篇（仍保留 Apple/Grace Hopper 統一記憶體列）。
- **「心法：最慢那段路」（1 頁）**：折成一句併入記憶體階層頁（第 7 頁）。
- **「CPU vs GPU」「GPU 解剖 SM/CUDA/Tensor」（2 頁）**：GPU 結構改由互動地圖①承擔（已含 SM → CUDA/Tensor core 第 5 層下鑽）；開場也從「兩個謎題」收回「一個謎題」（謎題②原由 CPU-vs-GPU 頁回收）。

## 互動地圖使用方式（第 4 頁指引）

- 開啟 `interactive/gpu-map.html`（任何瀏覽器、離線可用、單一檔案）。
- 操作：點發亮元件往內切；`Esc` 回上層；數字鍵 `1`–`5` 直接跳層；也可用 URL hash 直達（`#cluster` / `#node` / `#gpu` / `#sm` / `#core`）。
- 每層右側面板有頻寬數字表與「教學重點」，數字與合輯投影片一致（H100 世代約略值）。
- 建議講法：Cluster（跨節點最慢）→ Node（PCIe vs NVLink）→ GPU（HBM + ridge point）→ SM（warp / shared memory）→ **運算單元（CUDA core 純量逐格 vs Tensor core 整塊 tile，各做不一樣的事）**，講完按 `5` 停在運算單元，切回投影片接第 5 頁。

## 使用建議

- **完整講**：34 頁約 2.5 小時（含兩個互動地圖與 demo），適合工作坊。
- **精簡 keynote**：2、4（地圖①）、7–9（階層+價格+各家加速器比較）、10–15、18–21（decode 解謎 + prefetch 壓軸）、22–25（含互動②）、27–28（TPU/Groq）、32、34 約 70–85 分鐘。
- 講稿細節：本檔即唯一主講稿（原 `s1`–`s5` 單場講稿已併入後刪除）。撐場用的關鍵推導與現場 Q&A 見下方「## 關鍵推導速查」「## Q&A 速查」；Transformer 互動環節（第 25 頁）另見 [transformer-interactive.md](transformer-interactive.md)。

## 關鍵推導速查（被追問時用）

數字一律約略值（H100 世代）。縮寫見 [glossary.md](glossary.md)。

### 算術強度 / ridge point（Part 1–2）

- GEMM `C[M,N]=A·B`：FLOPs `= 2MKN`，理想 Bytes `= s·(MK+KN+MN)`（s = dtype 位元組）。
- 大方陣 `M=N=K=n`：AI `≈ 2n/(3s)` → 隨 n 線性增 → compute-bound。
- GEMV `M=1`：AI `≈ 2/s` → fp16 約 **1** → 永遠 memory-bound。
- ridge point `= 峰值算力 / 峰值頻寬`。H100：`990 TFLOPS / 3.35 TB/s ≈ 296 FLOPs/Byte`。
- 「990 TFLOPS / 3.35 TB/s 怎麼算」（第 14 頁）：算力 = 單元數 × 時脈 × 每 cycle 運算；頻寬 = 匯流排寬 × 時脈。tensor core 對 CUDA core 同時脈差約 15×。

### tiling 為什麼提高 AI（Part 2）

- naive matmul：每個輸出重讀整行整列 → HBM 讀 ~ O(N³)。
- tiled（tile 邊長 T）：T×T block 載入 shared memory 重複用 T 次 → HBM 讀 ~ O(N³/T)。
- AI ≈ O(N³)/O(N³/T) = **O(T)** → operating point 從 roofline 斜線往右推進平頂，受 shared memory 容量限。

### 藏延遲要多少 warp（Part 2）

- 藏住記憶體延遲 L 需約 `L / 每 warp 指令間隔` 個可切換 warp。HBM 延遲數百 cycle → 需數十 warp 駐留 → 這是 GPU 要「大量 thread」的根本原因。前提：有夠多 warp 且頻寬餵得上。

### decode 為何 memory-bound（Part 3，系列核心）

- 每步要讀 ≈ 權重（+ 當前 KV cache）；做的 FLOPs ≈ `2·params·batch`。
- AI ≈ `2·params·batch / (2·params)` = **batch**（fp16）。batch 小 → AI 小 → memory-bound。
- batch=1 單步延遲下限 = `權重位元組 / HBM 頻寬`。7B fp16：`14e9 / 3.35e12 ≈ 4.2 ms` → ~240 tok/s。這就是「<5% 利用率」的根源。
- KV cache 大小 ≈ `2 × layers × heads × head_dim × seq_len × batch × dtype_bytes`，隨 `seq_len × batch` 線性長大、每步都要讀。
- batch 攤平：單步延遲 ≈ `max(權重/頻寬, 2·params·batch/算力)`；batch 小被常數項主宰 → tokens/s 線性升；跨過 ridge（≈ 數百）後轉 compute-bound、趨平。throughput↑ ≠ latency↓。

### 資料搬遷 / overlap 上限（Part 4）

- 頻寬階梯：HBM ~TB/s ≫ NVLink ~900 GB/s（NVLink 4 雙向合計；每方向 450 GB/s，第四堂一律用每方向）≫ PCIe ~32–64 GB/s ≫ DRAM ~100 GB/s ≫ SSD ~7 GB/s。瓶頸 = 必經的最慢那段。
- pinned vs pageable：pageable 要先 staging 到 pinned bounce buffer（多一跳、不能 async）；pinned DMA 直達且可 `non_blocking` overlap，差約 2×。
- overlap 理想上限：搬一批 C、算一批 K。naive ≈ `N·(C+K)`，完美 overlap ≈ `C + N·max(C,K)`；`C≈K` 時加速 ≈ **2×**。`C≪K` 幫助小，`C≫K` 上限受 C 決定。

### 共同演化（Part 5）

- Amdahl：`S(N)=1/((1−p)+p/N)`。p=0.95、N=16896 → ≈ **20×** ≈ N=∞——N 夠大後瓶頸只剩 (1−p)。模型裡的「序列相依」就是 (1−p)：RNN 訓練的 T 步遞迴、transformer decode 的逐 token。
- RNN vs attention：同樣總 FLOPs，依賴圖一個是深度 T 的鏈（LSTM，T 步序列 + T 次 kernel launch），一個是深度 1 的寬層（attention，(B·T)×H 一次 GEMM）。
- FlashAttention：數學不變，tiling + 線上 softmax + 反向重算讓 T×T 矩陣**不落地 HBM** → bytes 大減、記憶體 O(T²)→O(T)、快 2–4×。T=8192/fp16 時單 head 的 S ≈ 134 MB 反覆進出 HBM。
- MQA/GQA：KV head 從「每 Q 一組」→「共用」。Llama2-70B GQA-8 把 KV bytes ÷8；T=4096/fp16、單條序列時 KV cache ~10.7 GB → ~1.3 GB（每 token 2.5 MB → 320 KB；省頻寬也省容量）。〔舊版寫 ~21 GB → ~2.6 GB，多算了 2×；320 KB/token 與第四堂第 19 頁的 Llama-3-70B 一致〕
- depthwise separable conv：FLOPs ≈ 標準的 1/8–1/9（C=256），但 depthwise 段 AI 個位數 → memory-bound，GPU 上 wall-clock 常只快 1.5–3×。「FLOPs ÷9 ≠ 速度 ÷9」。
- Mamba：遞迴 `h_t = Ā h_{t−1} + B̄ x_t`，但線性遞迴可寫成關聯運算 → **parallel scan** 在 O(log T) 深度算完 → 訓練平行；推論退回遞迴、每步 O(1) 狀態、無 KV cache 成長。

## Q&A 速查（現場準備）

- **加大 batch 不就解決了？** 對 throughput 有效（攤平權重讀取），對單一請求 latency 無幫助；且 batch 受 KV cache 容量限。
- **tiling/cache 不是能省頻寬？** 能——把資料留 shared memory 重複用 = 提高 AI、往 roofline 右推。
- **tensor core vs CUDA core？** tensor core 專做小矩陣乘加（如 16×16），吞吐遠高，但要餵對形狀與精度（fp16/bf16/fp8）→ 所以 kernel 要 tiling 把資料排好。
- **decode 為什麼不也用大 batch？** server 端會（continuous batching）；但單一使用者湊不到大 batch，且 KV cache 容量會先撐爆——這是 LLM 服務的核心工程問題。
- **speculative decoding / 量化算解法嗎？** 算。投機解碼用小模型猜、大模型一次驗多 token，攤平權重讀取；量化（int8/fp8）直接減少要搬的 bytes——都在打 memory-bound。
- **統一記憶體是不是就不用煩惱搬運？** 看機制：Apple 零複製確實不搬（但頻寬低）；NVIDIA UVM 仍會遷移（可能 thrash）；Grace Hopper 靠高速一致性互連把「搬」變便宜——三者別混為一談。
- **多卡一定要 NVLink 嗎？** 不一定，但卡間通訊量大的並行（tensor parallel、頻繁 all-reduce）在純 PCIe 上會被卡間頻寬拖死。
- **所以是硬體決定論嗎？** 不是死，是「延遲」。CNN 等了 23 年。讀論文要分清「想法輸在本質還是輸在當代硬體」——後者值得在硬體轉彎時重訪（遞迴架構正因推論成本復活）。
- **Transformer 的 O(T²) 不是比 RNN 的 O(T) 差？** 漸進複雜度是 CPU 思維。T 幾千時「O(T²) 但全平行、高 AI」實際遠勝「O(T) 但序列」；T 大到真的痛時社群也不回 RNN，而是 FlashAttention / 稀疏 / SSM 混合。
- **MoE 在這框架怎麼看？** 用「容量換 FLOPs」：參數多但每 token 只算少數 expert → 省 FLOPs，**不省 bytes/通訊**（權重仍要就位、all-to-all 路由）→ 又一個「FLOPs ≠ 速度」案例。
- **怎麼預判下一代架構？** 看硬體稀缺資源往哪移。現在稀缺是 HBM 頻寬/容量與互連（不是 FLOPs）→ 押「省 bytes」：激進量化、KV 壓縮、固定狀態遞迴混合、計算/通訊重疊。

## 承上啟下（銜接第二堂）

- **本堂留下的問題**：第 18–20 頁回答了「為什麼 decode 用不到算力」，但一直把 Transformer 當成一整塊在看——它的每個 block 到底跑在 GPU 的哪個單元？模型大到一張卡裝不下時又怎麼辦？
- **下一堂**（[第二堂](class2-transformer-gpu.md)）：Part A 把玩具 Transformer 逐 block 對到 tensor core / CUDA core；Part B 讓模型長大、溢出一顆 HBM，看資料平行的四個問題與 NVIDIA 互連。
- 收尾建議口白：「今天我們有了一把尺和一張表。下一堂，拿這把尺一格一格量 Transformer；量完之後，再看一張卡裝不下時會發生什麼。」
