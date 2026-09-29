# 術語與縮寫對照表（Glossary）

整個讀書會系列（四堂投影片、demo、互動地圖、各堂講稿）用到的縮寫、英文全稱、中文與一句話說明。
> 用法:第一次出現某縮寫時可回查這張表;括號內的 Part 對應 [合輯](full-series.md) 的篇章。
> 校準說明:系列已更迭到聚焦版合輯(硬體架構 × Transformer)。下表仍保留**全部**術語以利回查,但標 ⊘ 者其正文已從合輯移除(ASR 案例、NVLink/GPUDirect/UVM),只在 demo 或概念背景出現——詳見合輯「## 聚焦版移除的內容」。

---

## 1. 效能判讀(Part 1–2)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| Roofline | Roofline model | 屋頂線模型 | 把「達成算力」對「算術強度」畫出來,一眼看出被算力還是頻寬卡住 |
| AI | Arithmetic Intensity | 算術強度 | 完成運算的 FLOPs ÷ 需搬動的 Bytes(單位 FLOPs/Byte) |
| — | compute-bound | 算力受限 | 瓶頸是峰值算力,加大 batch 也快不了(AI 高) |
| — | memory-bound | 記憶體受限 | 瓶頸是記憶體頻寬,大多時間在等資料(AI 低) |
| — | ridge point | 轉折點 / 脊點 | 峰值算力 ÷ 峰值頻寬;AI 低於它就是 memory-bound(H100 ≈ 300) |
| — | latency / throughput | 延遲 / 吞吐 | 延遲=單一請求多快;吞吐=單位時間總量(batch 換的是後者) |
| — | latency-oriented / throughput-oriented | 延遲導向 / 吞吐導向 | CPU 為前者(少數強核)、GPU 為後者(海量簡單核) |
| Amdahl | Amdahl's Law | 阿姆達爾定律 | 加速上限 = 1/((1−p)+p/N);序列部分 (1−p) 是天花板 |

## 2. 運算與數值精度(Part 1–2、Transformer 硬體層)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| FLOP(s) | Floating-Point Operation(s) | 浮點運算(次數) | 一次乘或加算一個 FLOP;FLOPS=每秒 FLOP 數 |
| TFLOPS | Tera-FLOPS | 兆次浮點運算/秒 | 10¹² FLOPS;H100 BF16 tensor ≈ 990 TFLOPS |
| GEMM | GEneral Matrix-Matrix multiply | 通用矩陣×矩陣乘法 | 大矩陣乘大矩陣,AI 高 → compute-bound(訓練/prefill 主體) |
| GEMV | GEneral Matrix-Vector multiply | 通用矩陣×向量乘法 | 矩陣乘向量,AI≈1–2 → memory-bound(decode 主體) |
| FMA | Fused Multiply-Add | 融合乘加 | a×b+c 一條指令算完(算 2 FLOPs);CUDA core 的基本動作 |
| GELU | Gaussian Error Linear Unit | 高斯誤差線性單元 | 一種激活函數,Transformer FFN 常用(比 ReLU 更平滑);由 CUDA core 執行 |
| QKV | Query / Key / Value | 查詢鍵值(合稱) | attention 三個投影矩陣的合稱;QKV 投影是每層第一道大 GEMM |
| MMA | Matrix Multiply-Accumulate | 矩陣乘加 | tensor core 的基本動作:一個 tile(如 16×16)一拍算完 |
| MAC | Multiply-ACcumulate | 乘加單元 | systolic array / MXU 的基本格子 |
| im2col | image-to-column | 影像轉欄 | 把卷積攤平成大矩陣乘法的常見手法(CNN→GEMM) |
| tiling | tiling / blocking | 分塊 | 把資料切塊載入 SRAM 重複用,提高 AI、減少 HBM 讀取 |
| fp32 / fp16 | (single / half) floating point | 單精度 / 半精度 | 32-bit / 16-bit 浮點 |
| bf16 | brain floating point 16 | bfloat16 | 16-bit 但指數範圍同 fp32;TPU 帶進主流、訓練常用 |
| fp8 / fp4 | 8-bit / 4-bit floating point | 8 位元 / 4 位元浮點 | 更低精度,H100 Transformer Engine 用 fp8 |

## 3. GPU 微架構(Part 1、互動地圖)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| GPU | Graphics Processing Unit | 圖形處理器 | 這裡指做通用運算的 GPGPU |
| GPGPU | General-Purpose computing on GPU | 通用 GPU 運算 | 把 GPU 拿來做非繪圖的通用平行運算 |
| SM | Streaming Multiprocessor | 串流多處理器 | GPU 的基本運算單元;H100 有 132 個 |
| CUDA | Compute Unified Device Architecture | (NVIDIA 運算平台) | NVIDIA 的 GPU 運算架構與程式模型 |
| CUDA core | CUDA core | CUDA 核心 | 通用浮點/整數運算單元(一個 SM 128 個) |
| Tensor core | Tensor Core | 張量核心 | 矩陣乘加(MMA)專用單元,衝高算力 |
| SXM | Server eXtension Module | (伺服器擴充模組封裝) | NVIDIA 高功耗 GPU 的封裝形式(H100 SXM 有更高頻寬 NVLink);對應消費型的 PCIe 版 |
| DGX | DGX | (NVIDIA DGX 伺服器) | NVIDIA 整合多張高端 GPU + NVSwitch 的機架型伺服器節點(如 DGX H100 = 8× H100 SXM) |
| warp | warp | 線程束 | 32 條 thread 一組,一起執行同一指令 |
| SIMT | Single Instruction, Multiple Threads | 單指令多線程 | 一個指令同時驅動一個 warp 的 32 條 thread |
| lane | lane | 通道 | 一條運算車道;H100 ≈ 16,896 條 FP32 lane |
| thread | thread | 線程 / 執行緒 | 最小執行單位;GPU 靠「大量 thread」藏延遲 |
| register | register | 暫存器 | 每 thread 私有、晶片內最快一層 |
| latency hiding | latency hiding | 藏延遲 | 某 warp 等記憶體時切到別的 warp,讓 SM 不空等 |

## 4. 記憶體與階層(Part 1、Transformer 硬體層)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| SRAM | Static Random-Access Memory | 靜態隨機存取記憶體 | 晶片內快取的材料(register/L1/L2);快但小 |
| DRAM | Dynamic Random-Access Memory | 動態隨機存取記憶體 | 主記憶體材料(HBM/GDDR/DDR);慢但大 |
| HBM | High Bandwidth Memory | 高頻寬記憶體 | 3D 堆疊 DRAM + 超寬匯流排,資料中心 GPU 用(2–4.8 TB/s) |
| GDDR | Graphics DDR | 繪圖用 DDR 記憶體 | 消費卡用,較窄匯流排、約 HBM 的 1/3 頻寬 |
| DDR5 | Double Data Rate 5 | 第五代 DDR | CPU 主記憶體(~100 GB/s) |
| LPDDR | Low-Power DDR | 低功耗 DDR | Apple 統一記憶體 / Grace 用 |
| L1 / L2 | Level-1 / Level-2 cache | 一級 / 二級快取 | L1 每 SM(與 shared 共享)、L2 全 SM 共用(~50MB) |
| shared memory | shared memory | 共享記憶體 | 每 SM 的程式可控高速暫存(~228 KB),tiling 的槓桿 |
| global memory | global memory | 全域記憶體 | 指 GPU 的 HBM,整卡可見 |
| KV cache | Key-Value cache | 鍵值快取 | 快取過去 token 的 K、V,避免 decode 每步重算 attention |
| NVMe SSD | Non-Volatile Memory express SSD | 高速固態硬碟 | 資料集/權重來源(~7 GB/s) |

## 5. 互連與資料搬遷(Part 4)

> ⊘ 合輯聚焦版正文僅保留 PCIe + pinned/pageable + prefetch/overlap;**NVLink/NVSwitch/C2C、GDS、UVM** 已移除(方案表仍保留 Apple/Grace Hopper 統一記憶體列)。以下術語留作概念背景與 demo 04 參考。

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| PCIe | Peripheral Component Interconnect express | (主機↔裝置匯流排) | Host↔GPU 的橋(Gen5 ~64 GB/s),最常被跨越的瓶頸 |
| NVLink | NVLink | (NVIDIA 高速互連) | GPU↔GPU / CPU↔GPU 直連(~900 GB/s),比 PCIe 快一個量級 |
| NVSwitch | NVSwitch | (NVLink 交換器) | 節點內多卡(如 8 卡)全互連 |
| C2C | Chip-to-Chip | 晶片間互連 | NVLink-C2C 連 Grace(CPU)與 Hopper(GPU) |
| H2D / D2H | Host-to-Device / Device-to-Host | 主機→裝置 / 裝置→主機 | 跨 PCIe 的資料搬運方向 |
| DMA | Direct Memory Access | 直接記憶體存取 | 硬體直接搬資料、不經 CPU |
| pinned memory | pinned (page-locked) memory | 鎖頁記憶體 | 實體位址固定,DMA 直達且可 async,比 pageable 快 ~2× |
| pageable memory | pageable memory | 可換頁記憶體 | OS 可換頁,DMA 前要先複製到 pinned bounce buffer(多一跳) |
| bounce buffer | bounce buffer | 中繼緩衝區 | pageable 傳輸時暫存資料的固定緩衝 |
| GDS | GPUDirect Storage | (SSD 直達 GPU) | DMA 從 SSD 直達 HBM、繞過 CPU,適合大資料/權重載入 |
| UVM | Unified Virtual Memory | 統一虛擬記憶體 | NVIDIA「遷移式」單一指標,page fault 觸發頁面遷移 |
| overlap | compute/copy overlap | 計算與搬運重疊 | 用第二條 stream 預取下一批,把搬運藏在運算後面 |
| prefetch | prefetch | 預取 | 在需要前先把資料搬到對的地方 |
| ZeRO | Zero Redundancy Optimizer | 零冗餘優化器 | 把優化器狀態/梯度/權重切片散到多卡,省訓練記憶體 |

## 6. 訓練 / 推論流程(Part 3)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| training | training | 訓練 | 大 batch、大 GEMM、compute-bound;痛點常是「裝不下」(容量) |
| inference | inference | 推論 | 用訓好的模型產出;分 prefill 與 decode 兩階段 |
| prefill | prefill | 預填充 | 一次吃整段 prompt → 大 GEMM、compute-bound |
| decode | decode | 解碼 / 逐字生成 | 一次一 token、batch 小 → GEMV、memory-bound |
| autoregressive | autoregressive | 自迴歸 | 一次生一個、依賴前一個輸出(序列相依) |
| token | token | 詞元 | 模型處理的最小文字單位 |
| batch | batch (size) | 批次(大小) | 一次處理的樣本數;加大攤平權重讀取、換吞吐 |
| activations | activations | 啟動值 | 前向算出的中間結果;訓練要留著等 backward(吃容量) |
| backward | backward pass | 反向傳播 | 算梯度的反向計算,也是 GEMM |
| gradient | gradient | 梯度 | 反向算出的參數更新方向 |
| optimizer state | optimizer state | 優化器狀態 | Adam ≈ 2× 參數量(fp32),訓練容量大戶 |
| Adam | Adaptive Moment estimation | (一種優化器) | 常用優化器,維護一階/二階動量 |

## 7. Transformer / Attention 結構(Part 5、Transformer 互動地圖)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| Transformer | Transformer | (一種模型架構) | 用 attention 取代遞迴、為平行而生的架構 |
| attention | (self-)attention | (自)注意力 | token 之間交換資訊的機制;Q·Kᵀ→softmax→·V |
| Q / K / V | Query / Key / Value | 查詢 / 鍵 / 值 | attention 的三個投影矩陣 |
| MHA | Multi-Head Attention | 多頭注意力 | 把維度切成多個 head 各自算 attention(多一個平行軸) |
| head | (attention) head | 注意力頭 | 一份獨立的 Q/K/V 子空間;玩具地圖用 2 個 |
| d_model | model dimension | 模型維度 | 每個 token 的向量維度(玩具=6、真實 12288) |
| d_head | head dimension | 單頭維度 | d_model ÷ heads(玩具=3、真實 128) |
| FFN | Feed-Forward Network | 前饋網路 | 每個 token 獨立過的兩層 GEMM(token 軸天然平行) |
| MLP | Multi-Layer Perceptron | 多層感知器 | FFN 的別名 |
| LayerNorm | Layer Normalization | 層正規化 | 穩定訓練的正規化層 |
| residual | residual connection | 殘差連接 | 把輸入直接加到輸出(Add),幫助深層訓練 |
| causal mask | causal mask | 因果遮罩 | 擋住「看未來 token」的上三角遮罩 |
| softmax | softmax | 柔性最大值 | 把分數轉成機率分布 |
| positional encoding | positional encoding | 位置編碼 | 把 token 的順序資訊加進 embedding |
| embedding | embedding | 嵌入 | 把 token 映射成向量 |
| MQA | Multi-Query Attention | 多查詢注意力 | 所有 Q head 共用一組 KV → KV cache 大減 |
| GQA | Grouped-Query Attention | 分組查詢注意力 | 多個 Q head 共用一組 KV(MQA 與 MHA 折衷;Llama2-70B KV÷8) |
| FlashAttention | FlashAttention | (IO 感知 attention) | tiling + 線上 softmax,讓 T×T 矩陣不落地 HBM、省 bytes |

## 8. 模型架構(Part 5)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| CNN | Convolutional Neural Network | 卷積神經網路 | 對 pixel/channel/batch 全平行、im2col 後就是 GEMM |
| conv | convolution | 卷積 | CNN 的核心運算 |
| depthwise separable conv | depthwise separable convolution | 深度可分離卷積 | 把卷積拆成 depthwise+pointwise,省 FLOPs(但 GPU 上不一定快) |
| MobileNet / ConvNeXt | MobileNet / ConvNeXt | (兩個 CNN) | 前者為手機省 FLOPs、後者為 GPU 用大 kernel |
| RNN | Recurrent Neural Network | 遞迴神經網路 | token 軸序列相依(h_t 依賴 h_{t−1}),訓練難平行 |
| LSTM | Long Short-Term Memory | 長短期記憶網路 | 帶門控的 RNN |
| SSM | State Space Model | 狀態空間模型 | 訓練可平行掃描、推論退回遞迴的序列模型 |
| Mamba | Mamba | (一個 SSM) | selective SSM,decode 無 KV cache 成長 |
| Jamba / Griffin | Jamba / Griffin | (兩個混合架構) | attention 與 SSM/線性遞迴交錯的混血模型 |
| MoE | Mixture of Experts | 專家混合 | 參數多但每 token 只算少數 expert;省 FLOPs、不省 bytes/通訊 |
| parallel scan | parallel scan (prefix sum) | 平行掃描(前綴和) | 把線性遞迴在 O(log T) 深度內算完,讓 SSM 訓練可平行 |
| scaling law | scaling law | 規模法則 | 模型/資料/算力放大時效能可預測地提升 |

## 9. 語音辨識 ASR(Part 3)

> ⊘ 合輯聚焦版已移除 **ASR 案例**(Whisper vs CTC 對照,2 頁)——抽象的「decode memory-bound / 平行 vs 序列」改由 Transformer 互動地圖落地。以下術語留作 demo 03(asr_proxy)與概念背景參考。

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| ASR | Automatic Speech Recognition | 自動語音辨識 | 把語音轉文字;本系列的落地案例 |
| encoder / decoder | encoder / decoder | 編碼器 / 解碼器 | 前者吃輸入(可平行)、後者出文字(可能自迴歸) |
| Whisper | Whisper | (一個 ASR 模型) | attention encoder–decoder;decoder 自迴歸 → 延遲主因 |
| CTC | Connectionist Temporal Classification | 連結時序分類 | 一次輸出所有 frame 的字元機率、無自迴歸 → 推論快 |
| wav2vec2 | wav2vec 2.0 | (一個 ASR 模型) | encoder-only + CTC,高度平行 |
| Conformer | Conformer | (一個 ASR encoder) | 卷積(抓局部)+ attention(抓全局)的混合,ASR 主流 |
| frame | frame | 音框 | 語音切成的時間片段 |

## 10. 硬體方案 / 廠商(Part 4–5、互動地圖硬體層)

> 本類多為合輯**新增/強化**的主角:TPU(第 30 頁)、Groq LPU(第 31 頁)是聚焦版的硬體三選一重點;「統一記憶體」僅保留 Apple/Grace Hopper 列(NVIDIA UVM「遷移」機制的細節隨第 5 類一併淡出正文)。

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| TPU | Tensor Processing Unit | 張量處理器 | Google 的 GEMM 專用晶片,用脈動陣列 + 編譯器排程 |
| MXU | Matrix multiply Unit | 矩陣乘法單元 | TPU 的核心,128×128 個 MAC 排成脈動陣列 |
| systolic array | systolic array | 脈動陣列 | 權重釘住、資料一拍一拍流過的運算陣列(無動態排程) |
| weight-stationary | weight-stationary | 權重駐留(資料流) | 權重載入後不動,輸入流過去;systolic array 的一種資料流 |
| XLA | Accelerated Linear Algebra | (加速線性代數編譯器) | 把整個計算圖靜態編譯排程(TPU/JAX 用) |
| Groq | Groq | (一家 AI 晶片公司) | 主打推論、確定性執行,LPU 全 SRAM |
| LPU | Language Processing Unit | 語言處理器 | Groq 的晶片;~230MB 全 SRAM、~80 TB/s,砍掉 HBM |
| Grace Hopper | Grace Hopper (GH200) | (NVIDIA CPU+GPU) | Grace(CPU,LPDDR5X)+Hopper(GPU,HBM3)用 NVLink-C2C 連 |
| Transformer Engine | Transformer Engine | (H100 的功能) | 為 transformer 訓練做的 fp8 動態精度硬體 |
| 統一記憶體 | unified memory | unified memory | CPU/GPU 共用記憶體;NVIDIA=遷移、Apple=零複製、GH=一致性互連 |

## 11. 系統 / 平行(Part 1、Part 4)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| cluster | cluster | 叢集 | 多個節點用網路相連(gpu-map 最外層) |
| node | node | 節點 | 一台 server(如 2 CPU + 8 GPU) |
| IB | InfiniBand | (高速網路) | 節點間互連(~25–50 GB/s/link) |
| TP / PP / DP | Tensor / Pipeline / Data Parallel | 張量 / 管線 / 資料平行 | 三種把模型/資料切到多卡多節點的平行策略 |
| DataLoader | DataLoader | 資料載入器 | PyTorch 餵資料的元件;num_workers+pin_memory+prefetch 藏搬運 |
| kernel | (GPU) kernel | 核函式 | 在 GPU 上跑的一段平行程式 |
| stream | (CUDA) stream | 串流 | GPU 上的指令佇列;多條 stream 可重疊搬運與運算 |
| bank conflict | bank conflict | 記憶體庫衝突 | 多 thread 同時存取同一 shared memory bank 被序列化 |

## 12. NVIDIA 互連與 AI 儲存織物(互連地圖 / CMX)

> 對應互動地圖 [interactive/interconnect-map.html](../interactive/interconnect-map.html):從晶片到資料中心,資料走哪條協定。

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| scale-up | scale-up | 向上擴展 | 用記憶體語意(load/store、一致性)把多顆 GPU 綁成一顆大 GPU(NVLink 域內) |
| scale-out | scale-out | 向外擴展 | 用網路訊息語意(RDMA)把多台節點串成一台 AI factory |
| NVLink 5 | NVLink (5th gen) | (Blackwell 互連) | 每 GPU ~1.8 TB/s(18×100,雙向合計;每方向 900);NVLink 4(Hopper)~900 GB/s(雙向;每方向 450) |
| NVL72 | GB200 NVL72 | (機架級 scale-up 域) | 72 顆 Blackwell 當一個記憶體域,130 TB/s aggregate(NVSwitch fabric 上限 576 GPU/1 PB/s) |
| Blackwell | Blackwell (B200/GB200) | (NVIDIA GPU 世代) | Hopper 之後世代;NVLink 5、HBM3e |
| RDMA | Remote Direct Memory Access | 遠端直接記憶體存取 | 一台機器的 NIC 直接讀寫另一台的記憶體,CPU 不插手 |
| RoCE | RDMA over Converged Ethernet | (乙太網上的 RDMA) | 把 RDMA 跑在乙太網上(Spectrum-X 用 RoCEv2) |
| Spectrum-X | NVIDIA Spectrum-X | (AI 乙太網平台) | 為 AI 調過的乙太網;adaptive routing + 壅塞控制,~800 Gb/s |
| Quantum | NVIDIA Quantum InfiniBand | (IB 交換器) | InfiniBand 交換器;NDR 400 Gb/s / XDR 800 Gb/s |
| GPUDirect RDMA | GPUDirect RDMA | (網路直達 HBM) | 讓遠端 NIC 直接讀寫 GPU HBM、繞過 CPU |
| NVMe-oF | NVMe over Fabrics | (網路化 NVMe) | 把 NVMe 協定跑在網路上,遠端 flash 像本機般存取 |
| DPU | Data Processing Unit | 資料處理器 | 卸載網路/儲存/安全的處理器(NVIDIA = BlueField) |
| BlueField | NVIDIA BlueField | (DPU 產品) | NVIDIA 的 DPU;CMX 用 BlueField-4 當儲存處理器 |
| NCCL | NVIDIA Collective Communications Library | (集合通訊庫) | GPU 間 all-reduce/all-gather 等集合通訊,資料平行梯度同步靠它 |
| CMX | Context Memory (storage) | 情境記憶儲存 | 2026 NVIDIA 提出;把 KV cache 卸載到乙太掛載 flash 的 G3.5 層,~5× tok/s、能效 ~4×(第二堂投影片;另有 ~5× 說法,以官方頁為準) |
| G3.5 / context tier | context memory tier | 情境記憶層 | 記憶體階層新增的一層,夾在本機 SSD(G3)與共享儲存(G4)之間,專放 KV cache |
| DOCA Memos | NVIDIA DOCA Memos | (CMX 軟體 SDK) | 把 KV cache 當一等公民管理/分享/放置的 SDK,跑在 BlueField 上 |
| Dynamo | NVIDIA Dynamo | (分散式推論框架) | 協調 prefill/decode/KV cache,KV-aware 請求調度,整合 CMX |
| NIXL | NVIDIA Inference Transfer Library | (推論搬運庫) | 協調 KV 在各記憶體/儲存層間搬運,decode 前把 KV prestage 回 HBM |
| prestage | prestaging | 預先搬入 | 在 decode 之前把需要的 KV 從 CMX 搬回 HBM,不卡生成(prefetch 的一種) |

## 13. 推論引擎(第三堂)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| SGLang | Structured Generation Language | (推論引擎) | 前端 DSL + RadixAttention + cache-aware 排程;「替你猜工作負載」的那一家 |
| vLLM | vLLM | (推論引擎) | PagedAttention 起家;模型/硬體覆蓋最廣、後端可插拔 |
| PagedAttention / 分頁 KV | PagedAttention | 分頁式 KV | KV 切成固定大小 block(16 token)、按需配置,浪費從 60–80% 降到 <4%;兩家共同地基 |
| RadixAttention | RadixAttention (radix tree) | 基數樹前綴快取 | 用前綴樹索引已算過的 KV,任意 token 分叉點都能命中(SGLang) |
| APC | Automatic Prefix Caching | 自動前綴快取 | vLLM 以鏈式雜湊逐 block 查表,命中須對齊 block 邊界 |
| continuous batching | continuous / in-flight batching | 連續批次 | 排程單位從「一個請求」改成「一次 forward」,隨時插入/退出 |
| chunked prefill | chunked prefill | 分塊 prefill | 長 prompt 切塊、每步固定 token 預算混 decode;緩解 TTFT vs ITL 衝突 |
| TTFT | Time To First Token | 首字延遲 | 由 prefill(compute-bound)決定 |
| ITL / TPOT | Inter-Token Latency / Time Per Output Token | 字間延遲 | 由 decode(memory-bound)決定 |
| XGrammar | XGrammar | (結構化輸出後端) | 把 JSON schema/正則編成 FSM + 位元遮罩;SGLang、vLLM、TRT-LLM 預設後端 |
| FSM | Finite State Machine | 有限狀態機 | 結構化輸出的約束;屬於「請求狀態」,不可跨請求共享 |
| jump-forward | jump-forward decoding | 跳躍解碼 | 語法唯一確定時直接吐多個 token、不跑模型(SGLang) |
| CUDA Graph | CUDA Graph | (kernel 錄放) | 把一整串 kernel launch 錄成圖、之後 replay,省 CPU 開銷 |
| 投機解碼 | speculative decoding | 投機解碼 | 小模型/草稿先猜 k 個、大模型一次驗;權重讀 1 次換 k 倍 FLOPs(AI 從 1 變 k) |
| MTP | Multi-Token Prediction | 多 token 預測 | 訓練時多預測幾步;推論時那些 head 當投機解碼的 draft(第四堂旋鈕④) |

## 14. 多機推論服務(第四堂後半)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| EP | Expert Parallelism | 專家平行 | 把 MoE 專家散到多卡;每層兩次 all-to-all(dispatch + combine) |
| dispatch / combine | dispatch / combine | 分派 / 匯回 | token 送去專家所在的卡 / 算完送回原卡;DeepSeek-V3 每字 58 層 × 2 = 116 次 |
| EPLB | Expert Parallelism Load Balancer | 專家負載均衡器 | 熱門專家做副本、定期重排;專家熱點＝資料傾斜 |
| DeepEP | DeepEP | (all-to-all 通訊庫) | DeepSeek 開源;normal 模式(prefill)與 low-latency 模式(decode,純 RDMA、可進 CUDA Graph) |
| PD 分離 | prefill–decode disaggregation | 預填/解碼分離 | prefill 與 decode 放不同機器池、各自最佳配置,中間傳 KV |
| KV 交接 | KV transfer | KV 傳輸 | PD 分離時 KV 從 prefill 機搬到 decode 機(SGLang 推、vLLM+NIXL 拉) |
| DP attention | data-parallel attention | 資料平行注意力 | attention 各卡只算自己那批請求、KV 不重複(MLA 無法按 head 切時特別重要) |
| TBO | Two-Batch Overlap | 雙批重疊 | 一批切兩個 micro-batch,一個通訊時另一個計算 |
| cache-aware router | cache-aware load balancing | 快取感知路由 | 同時看前綴命中與各機負載來分流;router 的快取視圖是近似的 |
| Mooncake | Mooncake | (Kimi 的 KV 中心架構) | Moonshot 以 KVCache 為中心的分離式服務架構;SGLang 的 KV 傳輸後端之一 |
| straggler | straggler | 拖後腿節點 | 同步集合通訊下最慢那張卡決定整體速度;「慢了」與「掛了」難分 |
| NVSHMEM / IBGDA | NVSHMEM / InfiniBand GPUDirect Async | (GPU 發起的 RDMA) | GPU 直接敲網卡門鈴、不經 CPU;DeepEP low-latency 模式的基礎 |
| scale-up 域 | scale-up domain | 高速互連域 | 一組用 NVLink 直連的卡(H100/B200 機 = 8、NVL72 = 72);第四堂的真瓶頸 |

## 15. 模型端的效率旋鈕(第四堂前半)

| 縮寫 / 術語 | 英文全稱 | 中文 | 一句話說明 |
|---|---|---|---|
| MLA | Multi-head Latent Attention | 多頭潛在注意力 | K/V 壓成低秩 latent 再存;DeepSeek-V3 ≈ 70 KB/token(同規模 MHA ~4 MB) |
| MoE | Mixture of Experts | 混合專家 | 每 token 只過少數專家;省 FLOPs 與權重讀取、單卡不省容量 |
| 活躍參數 | active parameters | 活躍參數 | 每 token 實際用到的參數量(如 671B-A37B 的 37B);決定 decode 速度 |
| DSA / CSA / MSA | DeepSeek / Compressed / MiniMax Sparse Attention | 稀疏注意力 | 保留完整 KV、每個 query 只看 top-k;KV 還在 → prefix caching/投機解碼仍可用 |
| 線性注意力 | linear attention (Lightning / GDN / KDA) | 線性注意力 | 不存 KV、改固定大小遞迴狀態;打壞 KV 量化、前綴快取、投機解碼三個生產系統 |
| QAT | Quantization-Aware Training | 量化感知訓練 | 訓練時就模擬低精度;Kimi K3「出廠就是 4-bit」 |
| MXFP4 / NVFP4 | Microscaling FP4 / NVIDIA FP4 | (4-bit 浮點) | 區塊共享縮放的 4-bit 格式,對齊 Blackwell FP4 tensor core |

---

## 速記:最常混淆的幾組

- **GEMM vs GEMV**:矩陣×矩陣(compute-bound,訓練/prefill)vs 矩陣×向量(memory-bound,decode)。**速度差別的核心**。
- **SRAM vs DRAM**:晶片內快取材料(快小)vs 主記憶體材料(慢大);HBM 是一種 DRAM。**KV cache 因為太大塞不進 SRAM,只能住 HBM、每步串流**。
- **FMA vs MMA**:CUDA core 的純量乘加 vs tensor core 的一整塊 tile 乘加。
- **MHA / MQA / GQA**:KV head 數從「每個 Q 一組」→「全部共用一組」→「分組共用」,換 KV cache 頻寬。
- **HBM / NVLink / PCIe**:TB/s / ~900 GB/s–1.8 TB/s / ~64 GB/s,差一個量級往下掉,瓶頸常在 PCIe。
- **遷移 / 零複製 / 一致性互連**:三種「unified memory」機制(NVIDIA UVM / Apple / Grace Hopper)。
- **scale-up vs scale-out**:NVLink 把多顆 GPU 當一顆(記憶體語意)vs 網路把多台串起來(訊息語意);頻寬差一個量級——張量平行留 scale-up、資料平行才跨 scale-out。
- **CMX / G3.5**:KV cache 是「短命、可重算」的 AI-native 資料,CMX 給它專屬一層(BlueField-4 + Spectrum-X flash),省去硬塞 HBM 或重算 prefill(呼應 Part 3 decode memory-bound、Part 4 prefetch)。

> 數字一律約略值(以 H100 世代為主),詳見 [合輯](full-series.md) 與各場講稿。
