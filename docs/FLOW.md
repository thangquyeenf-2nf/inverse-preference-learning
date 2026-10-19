# Flow hoá: Phương pháp luận, Thuật toán & Quá trình huấn luyện IPL

Tài liệu này mô tả bằng sơ đồ (Mermaid) toàn bộ luồng của **Inverse Preference Learning (IPL)**
như được hiện thực trong repository này, ánh xạ trực tiếp tới mã nguồn.

| Thành phần | File |
|---|---|
| Điểm vào huấn luyện | `scripts/train.py` |
| Config → model/trainer | `research/utils/config.py` |
| Vòng lặp huấn luyện | `research/utils/trainer.py` |
| IPL + IQL (thuật toán chính) | `research/algs/ipl_iql.py` |
| IPL + AWAC | `research/algs/ipl_awac.py` |
| Baseline Preference-IQL (MR / Oracle) | `research/algs/pref_iql.py` |
| Dữ liệu preference + replay | `research/datasets/feedback_buffer.py` |
| Thu thập dữ liệu online | `research/algs/off_policy_algorithm.py` |

---

## 1. Phương pháp luận

### 1.1. Bài toán

Preference-based RL: không có hàm thưởng `r(s,a)`. Thay vào đó có tập dữ liệu so sánh cặp
`(σ¹, σ², y)` — hai đoạn quỹ đạo dài `k` bước và nhãn `y ∈ {0, 1}` cho biết người dùng thích đoạn nào hơn.
Mô hình người dùng được giả định theo **Bradley–Terry**:

```
P[σ² ≻ σ¹] = sigmoid( Σ_t r(s²_t, a²_t) − Σ_t r(s¹_t, a¹_t) )
```

### 1.2. So sánh: PbRL truyền thống vs. IPL

```mermaid
flowchart LR
    subgraph TRAD["PbRL truyền thống: MR-IQL / PEBBLE"]
        direction TB
        P1["Dữ liệu preference<br/>(σ¹, σ², y)"] --> R1["Học reward model r_ψ<br/>loss BCE Bradley–Terry"]
        R1 --> R2["Gán nhãn reward<br/>cho replay data"]
        R2 --> Q1["Học Q bằng Bellman:<br/>Q ← r_ψ + γV"]
        Q1 --> PI1["Trích xuất policy π"]
    end

    subgraph IPL["IPL: không có reward network"]
        direction TB
        P2["Dữ liệu preference<br/>(σ¹, σ², y)"] --> IB["Inverse Bellman:<br/>r_Q = Q(s,a) − γ·V(s')"]
        IB --> BT["Bradley–Terry trên r_Q<br/>loss BCE"]
        BT --> REG["+ Chính quy hoá χ²<br/>λ·E[r_Q²]"]
        REG --> Q2["Cập nhật trực tiếp Q"]
        Q2 --> PI2["Trích xuất policy π<br/>(IQL / AWAC)"]
    end
```

Ý tưởng cốt lõi: với một policy cố định, **toán tử Bellman ngược**
`(T* Q)(s,a) = Q(s,a) − γ E_{s'}[V(s')]` cho phép biểu diễn hàm thưởng hoàn toàn thông qua Q.
Do đó ta có thể thay `r` trong mô hình Bradley–Terry bằng `Q − γV` và tối ưu **trực tiếp Q** từ preference,
bỏ hẳn bước học reward model và giảm số tham số.

### 1.3. Hàm mục tiêu tổng quát

```mermaid
flowchart TB
    Q["Q-function Q_θ<br/>ensemble E=2"] --> R["r_Q(s,a) = Q_θ(s,a) − γ·V(s')"]
    V["Value V_φ(s')<br/>(IQL) hoặc Q(s', π(s')) (AWAC)"] --> R
    R --> SUM["Tổng theo đoạn:<br/>S¹ = Σ r_Q(σ¹), S² = Σ r_Q(σ²)"]
    SUM --> LOGIT["logit = S² − S¹"]
    LOGIT --> LPREF["L_pref = BCE(logit, y)"]
    R --> LCHI["L_χ² = λ · E[r_Q²]<br/>trên cả feedback + replay data"]
    LPREF --> LQ["L_Q = L_pref + L_χ²"]
    LCHI --> LQ
```

Vai trò của chính quy hoá χ² (`chi2_coeff = λ`): giữ cho reward ngầm định `r_Q` bị chặn,
tránh Q trôi tự do vì preference chỉ xác định reward đến một hằng số. Khi bật `target_clipping`,
`V(s')` được kẹp trong `[−1/(λ(1−γ)), 1/(λ(1−γ))]` cho phù hợp với miền giá trị mà χ² ngụ ý.

---

## 2. Thuật toán ứng dụng

### 2.1. Cấu trúc mạng (`ActorCriticValuePolicy`)

```mermaid
flowchart LR
    OBS["obs s"] --> ENC["Encoder<br/>(identity / MLP)"]
    ENC --> ACT["Actor π<br/>DiagonalGaussianMLPActor"]
    ENC --> CRI["Critic Q<br/>ContinuousMLPCritic, ensemble=2"]
    ENC --> VAL["Value V<br/>MLPValue, ensemble=1<br/>(chỉ IPL-IQL)"]
    CRI -. "Polyak τ=0.005" .-> TGT["Target network"]
```

Ba optimizer Adam riêng biệt: `actor`, `critic` (gồm cả encoder), `value`.

### 2.2. IPL-IQL — một bước `train_step`

```mermaid
flowchart TB
    START(["train_step(batch, step)"]) --> SPLIT["Tách batch = (replay_batch, feedback_batch)"]
    SPLIT --> FBCHK{"Đến kỳ thu feedback?<br/>step − last ≡ 0 mod reward_freq<br/>và total_feedback < max_feedback"}
    FBCHK -- Có --> COLLECT["_collect_feedback()<br/>xem mục 2.4"]
    FBCHK -- Không --> HASFB
    COLLECT --> HASFB{"feedback_batch tồn tại?"}
    HASFB -- Không --> END0(["return metrics"])
    HASFB -- Có --> BUILD["Ghép batch lớn:<br/>obs, action, next_obs từ σ¹ | σ² | replay<br/>split = [B·S, B·S, B_r]"]

    BUILD --> ENCODE["obs → encoder<br/>next_obs → encoder (no grad)"]

    ENCODE --> VSTEP["① Cập nhật Value<br/>target_q = min_E Q_target(s,a)<br/>L_V = expectile(τ=0.7)(target_q − V(s))"]
    VSTEP --> VW["Cân bằng feedback/replay<br/>theo value_replay_weight"]
    VW --> VOPT["optim['value'].step()"]

    VOPT --> RSTEP["② Reward ngầm định<br/>r = Q(s,a) − γ·clip(V(s'))"]
    RSTEP --> CHUNK["Tách r → r¹, r², r_replay<br/>reshape (E, B, S)"]
    CHUNK --> LOGIT["logit = Σ_S r² − Σ_S r¹"]
    LOGIT --> QLOSS["L_pref = BCEWithLogits(logit, label)"]
    CHUNK --> CHI["L_χ² = λ·[(1−w)·mean(r¹²,r²²) + w·mean(r_replay²)]"]
    QLOSS --> QOPT["optim['critic'].step() trên L_pref + L_χ²"]
    CHI --> QOPT

    QOPT --> ASTEP["③ Cập nhật Actor (AWR)<br/>adv = target_q − V(s)<br/>weight = min(exp(adv/β), 100)<br/>L_π = weight · (−log π(a|s))"]
    ASTEP --> AOPT["optim['actor'].step()"]

    AOPT --> TGT{"step % target_freq == 0?"}
    TGT -- Có --> POLYAK["θ' ← τθ + (1−τ)θ'"]
    TGT -- Không --> END1
    POLYAK --> END1(["return metrics:<br/>q_loss, chi2_loss, actor_loss, reward, q"])
```

### 2.3. IPL-AWAC — khác biệt so với IPL-IQL

```mermaid
flowchart LR
    subgraph IQL["IPL-IQL"]
        A1["V(s') = V_φ(s')<br/>học bằng expectile"]
        A2["adv = min Q_target − V_φ(s)"]
    end
    subgraph AWAC["IPL-AWAC"]
        B1["V(s') = Q_target(s', a'),<br/>a' ~ π(·|s')<br/>min hoặc mean qua ensemble"]
        B2["adv = Q(s,a) − Q(s, a_π)"]
    end
    A1 -. "không có mạng V riêng" .-> B1
    A2 -. "baseline lấy từ policy" .-> B2
```

Phần còn lại (BCE preference + χ², actor dạng advantage-weighted regression, Polyak target) giống hệt.

### 2.4. Thu thập feedback online (khi `max_feedback > 0`)

```mermaid
flowchart TB
    S(["_collect_feedback(step)"]) --> SIZE["batch = init_feedback_size<br/>(constant) hoặc giảm tuyến tính (linear)<br/>batch = min(batch, ngân sách còn lại)"]
    SIZE --> MODE{"total_feedback <<br/>num_uniform_feedback?"}
    MODE -- Có --> UNI["Lấy ngẫu nhiên batch cặp đoạn<br/>từ replay buffer"]
    MODE -- Không --> CAND["Lấy batch × feedback_sample_multiplier<br/>cặp ứng viên"]
    CAND --> ENS["Với mỗi thành viên ensemble Q:<br/>r = Q − γV(s')<br/>P = sigmoid(Σr² − Σr¹)"]
    ENS --> DIS["disagreement = std_E(P)"]
    DIS --> TOPK["Chọn top-k bất đồng nhất"]
    UNI --> LABEL["Gán nhãn oracle:<br/>y = 1[return¹ < return²]"]
    TOPK --> LABEL
    LABEL --> ADD["feedback_dataset.add(queries, labels)<br/>total_feedback += k"]
```

### 2.5. Baseline Preference-IQL (MR-IQL / Oracle-IQL) để đối chiếu

```mermaid
flowchart LR
    A["step < reward_steps"] -->|Giai đoạn 1| RW["Học reward net r_ψ<br/>BCE trên preference"]
    A -->|Giai đoạn 2| IQL["IQL chuẩn:<br/>Q ← r_ψ(s,a) + γV(s')<br/>V ← expectile, π ← AWR"]
```

---

## 3. Quá trình huấn luyện

### 3.1. Từ dòng lệnh đến vòng lặp

```mermaid
flowchart TB
    CLI["python scripts/train.py --config X.yaml --path out/"] --> LOAD["Config.load(yaml)"]
    LOAD --> WANDB["(tuỳ chọn) wandb.init nếu có WANDB_API_KEY"]
    WANDB --> SAVE["Lưu config.yaml + git_hash.txt vào out/"]
    SAVE --> PARSE["config.parse() — resolve tên lớp"]
    PARSE --> MODEL["config.get_model()<br/>→ IPL_IQL / IPL_AWAC / PreferenceIQL ..."]
    PARSE --> TRAINER["config.get_trainer()"]
    MODEL --> SETUP["Algorithm setup:<br/>processor, network + target,<br/>optimizers, dataset"]
    TRAINER --> TRAIN["trainer.set_model(model)<br/>trainer.train(path)"]
    SETUP --> TRAIN
```

Sweep `.json` (ví dụ `configs/gym_locomotion/locomotion.json`) chạy nhiều config/seed qua `tools/run_local.py` hoặc `tools/run_slurm.py`.

### 3.2. Pipeline dữ liệu (`ReplayAndFeedbackBuffer`)

```mermaid
flowchart LR
    subgraph REPLAY["replay_class"]
        D4["D4RLDataset<br/>(locomotion)"]
        RM["RobomimicDataset"]
        EMP["EmptyDataset<br/>(MetaWorld: chỉ preference)"]
        RB["ReplayBuffer<br/>(online)"]
    end
    subgraph FB["feedback_class: PairwiseComparisonDataset"]
        NPZ["*.npz: obs_1, obs_2,<br/>action_1, action_2, label"]
        SUB["Cắt ngẫu nhiên đoạn con<br/>subsample_size từ segment_size"]
        NPZ --> SUB
    end
    REPLAY --> ZIP["Mỗi bước yield<br/>(replay_batch, feedback_batch)"]
    SUB --> ZIP
    ZIP --> DL["torch DataLoader<br/>(batch_size=None)"]
    DL --> STEP["Algorithm.train_step"]
```

### 3.3. Vòng lặp `Trainer.train`

```mermaid
flowchart TB
    INIT(["Khởi tạo: optimizers, schedulers,<br/>logger, resume checkpoint nếu có"]) --> LOOP{"step < total_steps?"}
    LOOP -- Không --> DONE(["Kết thúc"])
    LOOP -- Có --> EXTRA["model.train_extras()<br/>online: tương tác env, thêm vào replay<br/>(random_steps đầu dùng hành động ngẫu nhiên)"]
    EXTRA --> BATCH["Lấy batch từ DataLoader"]
    BATCH --> TS["model.train_step(batch, step)<br/>xem mục 2.2"]
    TS --> SCHED["scheduler.step()"]
    SCHED --> LOG{"step % log_freq == 0?"}
    LOG -- Có --> WLOG["Ghi metric train/*:<br/>q_loss, chi2_loss, actor_loss, reward"]
    LOG -- Không --> EV
    WLOG --> EV{"step − last_eval ≥ eval_freq?"}
    EV -- Có --> EVAL["model.eval()<br/>validate() + eval_policy(num_ep)<br/>ghi eval/reward, eval/success"]
    EVAL --> BEST{"loss_metric tốt hơn?"}
    BEST -- Có --> SBEST["Lưu best_model"]
    BEST -- Không --> SFINAL
    SBEST --> SFINAL["Lưu final_model<br/>(ghi đè mỗi kỳ eval)"]
    SFINAL --> TRAINMODE["model.train()"]
    EV -- Không --> CK
    TRAINMODE --> CK{"step % checkpoint_freq == 0?"}
    CK -- Có --> SCK["Lưu checkpoint model_{step}"]
    CK -- Không --> INC
    SCK --> INC["step += 1"]
    INC --> LOOP
```

### 3.4. Siêu tham số chính theo benchmark

| Tham số | Gym locomotion (`configs/gym_locomotion/ipl_iql.yaml`) | MetaWorld (`configs/metaworld/ipl_iql.yaml`) |
|---|---|---|
| `chi2_coeff` λ | 0.5 | 3 |
| `expectile` | 0.7 | 0.75 |
| `beta` (nhiệt độ AWR) | 1/3 | 1/3 |
| `target_clipping` | True | False |
| `*_replay_weight` | 0.5 | 0.5 |
| Hidden layers | [256, 256] | [256, 256, 256] |
| Replay data | D4RL (batch 256) | không có (`EmptyDataset`) |
| Feedback batch / segment / subsample | 8 / 100 / 64 | 16 / 25 / 16 |
| `total_steps` | 1 000 000 | 200 000 |
| `eval_freq` / số episode | 5000 / 10 | 2500 / 20 |
| Optimizer | Adam, lr 3e-4 | Adam, lr 3e-4 |

---

## 4. Tóm tắt toàn trình

```mermaid
flowchart LR
    DATA["Preference pairs<br/>+ (tuỳ chọn) dữ liệu offline không nhãn"] --> IPL["IPL train_step<br/>① V expectile<br/>② Q ← BCE(Σ(Q−γV)) + χ²<br/>③ π ← AWR"]
    IPL --> TGT["Polyak target"]
    TGT --> IPL
    IPL --> EVAL["Đánh giá trên env<br/>(reward / success rate)"]
    EVAL --> CKPT["best_model / checkpoints"]
    IPL -. "online: disagreement<br/>query selection" .-> DATA
```
