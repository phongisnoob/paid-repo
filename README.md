# PAID — bản dựng lại từ đầu

Tái hiện "Phoneme-Aware Pronunciation Representations for L2-English L1-Background Accent
Identification" (PAID) trên L2-ARCTIC, 4-fold speaker-disjoint, kèm các thí nghiệm kiểm soát
mà bài báo còn thiếu.

## Trạng thái kiểm thử

| Phần                                                     | Trạng thái                                                        |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| Parser TextGrid, bộ phone ARPAbet-39, chia fold          | Đã test (33 test, `pytest tests`)                                 |
| Span pooling, masking, model, training loop, report      | Đã test trên dữ liệu giả và đã chạy trên dữ liệu thật             |
| Luồng wav + TextGrid -> cache (`extract_features.py`)    | Đã chạy trên L2-ARCTIC thật (Whisper-small)                       |
| Wrapper Whisper-small (`paid/encoders.py`)               | Đã chạy                                                           |
| Wrapper WavLM Base+ (`paid/encoders.py`)                 | Chưa chạy                                                         |
| MFA trên L2-ARCTIC thật                                  | Đã chạy, căn chỉnh được 26.867/26.867 câu (100%)                  |
| Con số 81.41% của bài báo                                | Chưa tái hiện: `paid` đạt 77.60 ± 8.64% (xem "Kết quả")           |

Môi trường chạy: <GPU>, Python <x.y>, torch <x.y.z>, transformers <x.y.z>, MFA <3.x>,
commit `<hash>`, ngày <dd/mm/yyyy>.

## Kết quả

Whisper-small, 4-fold speaker-disjoint, 10 epoch, một seed. Mean ± std theo fold (ddof=0):

| Biến thể         | fold0 | fold1 | fold2 | fold3 | Mean ± std   |
| ---------------- | ----- | ----- | ----- | ----- | ------------ |
| `utt`            | 88.89 | 78.66 | 85.11 | 68.49 | 80.29 ± 7.73 |
| `speechutt_proj` | 87.42 | 77.66 | 82.57 | 64.74 | 78.10 ± 8.45 |
| `paid_nopid`     | 87.40 | 76.06 | 81.89 | 65.74 | 77.77 ± 8.02 |
| `paid`           | 87.91 | 76.18 | 81.82 | 64.47 | 77.60 ± 8.64 |
| `paid_mlp_nopid` | 87.76 | 78.46 | 82.14 | 61.45 | 77.46 ± 9.82 |
| `paid_mlp`       | 85.27 | 76.53 | 82.54 | 64.28 | 77.16 ± 8.08 |
| `paid_tf`        | 81.74 | 77.29 | 80.26 | 63.26 | 75.64 ± 7.32 |
| `paid_tf_nopid`  | 80.17 | 78.08 | 80.72 | 60.73 | 74.92 ± 8.25 |

Bài báo: PAID 81.41%. Bản dựng này **không tái hiện được** con số đó (`paid` thấp hơn 3.8 điểm).

Nhận xét:
- Phone ID không cho lợi ích đo được ở cả ba loại head: `paid` - `paid_nopid` là -0.17 (linear),
  -0.30 (MLP), +0.72 (Transformer), dấu không nhất quán giữa các fold.
- `paid_tf` so với `paid_tf_nopid`: +0.68 điểm, CI 95% [-3.21, +4.82] (bootstrap theo 24 speaker).
  Khoảng này rộng nên không loại trừ được lợi ích hay tác hại cỡ vài điểm; kết quả chỉ cho thấy
  chưa có bằng chứng phone ID giúp ích.
- `paid` so với `speechutt_proj`: -0.49 điểm, CI 95% [-2.06, +1.22] (bootstrap theo speaker);
  span grouping và phone ID cộng lại không cho lợi ích đo được.
- `utt` so với `speechutt_proj`: +2.21 điểm, CI [+0.05, +4.54]; thêm projection làm kết quả kém đi.
- Mô hình càng nhiều tham số càng thấp (utt > projection > MLP > Transformer). Có thể do
  quá khớp trên 18 speaker huấn luyện mỗi fold, nhưng chưa được kiểm chứng.
- Fold 3 thấp (61-68%) do ba speaker khó: TXHC (L1 Chinese, thấp nhất), ZHAA và NJS. TXHC bị
  đoán lẫn giữa Chinese, Korean và Vietnamese ở `paid` và `utt`; alignment MFA của TXHC bình thường.
- Mỗi biến thể chỉ chạy một lần; chênh lệch dưới khoảng 1 điểm nằm trong nhiễu huấn luyện.
- Các control `*_shuffle_pid` chưa chạy vì phone ID chưa cho lợi ích nào cần kiểm tra thêm.
- WavLM chưa chạy.

## Cài đặt

```bash
pip install -r requirements.txt
conda create -n mfa -c conda-forge montreal-forced-aligner   # MFA 3.x, môi trường riêng
pytest tests -q                                             # kiểm tra repo trước khi dùng
```

## Quy trình

**1. Kiểm tra cấu trúc dữ liệu.** Code giả định mỗi speaker có `<ROOT>/<SPK>/wav/*.wav` và
`<ROOT>/<SPK>/transcript/*.txt`. Nếu thư mục của bạn khác, sửa `scripts/prepare_mfa.py`.
Tên speaker nằm trong `paid/l2arctic.py`; hãy đối chiếu với thư mục thực tế.

**2. Chuẩn bị corpus cho MFA**
```bash
python scripts/prepare_mfa.py --root /data/L2-ARCTIC --out mfa_corpus
```

**3. Forced alignment** (trong env `mfa`)
```bash
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa
mfa align mfa_corpus english_us_arpa english_us_arpa mfa_out -j 8 --clean --beam 100 --retry_beam 400
```
Xem `mfa_out/<SPK>/*.TextGrid`. Mở thử một file: cần có tier `phones` với nhãn dạng `AH0`.
Số file ít hơn số wav nghĩa là MFA không căn chỉnh được một số câu; `extract_features.py`
sẽ báo số câu bị bỏ qua.

**4. Trích đặc trưng (cần GPU)**
```bash
python scripts/extract_features.py --enc whisper --wav_root /data/L2-ARCTIC --mfa_out mfa_out --out feats/whisper
python scripts/extract_features.py --enc wavlm   --wav_root /data/L2-ARCTIC --mfa_out mfa_out --out feats/wavlm
```
Dòng `first utterance: frames (T, 768)` là điểm kiểm tra: `T x 0.02` phải xấp xỉ độ dài audio,
và `last phone ends at` phải nhỏ hơn hoặc xấp xỉ giá trị đó. Nếu lệch nhiều thì dừng lại,
đừng huấn luyện. Thêm `--keep_frames` (khoảng 7 GB) chỉ khi cần baseline `frameattn`.

**5. Tái hiện đúng bài báo**
```bash
python scripts/train.py --feat_dir feats/whisper --out runs/whisper --variants paid paid_nopid utt
python scripts/report.py runs/whisper
```

**6. Các control mà bài báo thiếu**
```bash
python scripts/train.py --feat_dir feats/whisper --out runs/whisper \
    --variants utt_proj speechutt speechutt_proj paid_shuffle_pid paid_mlp paid_tf \
               paid_mlp_nopid paid_tf_nopid paid_mlp_shuffle_pid paid_tf_shuffle_pid
python scripts/report.py runs/whisper
python scripts/report.py runs/whisper --compare paid utt          # bootstrap theo speaker
python scripts/report.py runs/whisper --per_speaker paid
```
Lặp lại với `feats/wavlm` (thêm `paid_nopid`, `frameattn`) để dựng lại Bảng I (chưa chạy).

Đã chạy: `speechutt_proj`, `paid_mlp`, `paid_mlp_nopid`, `paid_tf`, `paid_tf_nopid`.
Chưa chạy: `utt_proj`, `speechutt`, các biến thể `*_shuffle_pid`.

## Cách đọc kết quả

- `paid` so với `speechutt_proj`: khác nhau chỉ ở span grouping và phone ID. Nếu hai số gần
  nhau thì mức cải thiện của PAID đến từ việc bỏ silence và thêm projection, không phải từ
  cấu trúc phoneme.
- `paid` so với `paid_nopid`, và `paid_mlp` so với `paid_mlp_nopid`: đóng góp thật của phone ID.
- `paid_shuffle_pid` luôn gần bằng `paid`. Đây không phải lỗi: với mean-pool + linear thì
  hoán vị phone ID giữa các token không đổi logit (xem `tests/test_models.py`). Control này chỉ
  có ý nghĩa với `paid_mlp_shuffle_pid` và `paid_tf_shuffle_pid`.
- `report.py --compare` cho khoảng tin cậy bootstrap theo 24 speaker. Nếu khoảng này chứa 0
  thì chưa kết luận được hệ nào tốt hơn.

## Các quyết định mà bài báo không nêu (đã chọn, có thể đổi)

| Vấn đề | Lựa chọn trong code |
|---|---|
| Speaker nào thuộc fold nào | Speaker thứ k của mỗi nhóm L1 theo thứ tự trong `l2arctic.py`. Con số từng fold sẽ không khớp bài |
| Timestamp frame | Tâm frame, `(t + 0.5) x 0.02 s` |
| Span trong Eq. (3) | Nửa mở `[s, e)` để frame biên không bị đếm hai lần |
| Span ngắn hơn 1 frame | Lấy frame gần tâm span nhất (Eq. 4 sẽ chia cho 0) |
| Whisper padding 30 s | Cắt về độ dài thật, `ceil(n_samples / 320)` frame |
| `Whisper-Utt` baseline | Mean trên mọi frame thật, head là một lớp Linear, không chuẩn hóa |
| Std qua các fold | Độ lệch chuẩn tổng thể (`ddof=0`) |
| Chuẩn hóa đặc trưng đầu vào | Không có (bài chỉ nêu LayerNorm sau projection) |
| Validation | Không có, cố định 10 epoch như bài |

## Cấu trúc

```
paid/            phones, l2arctic (fold), textgrid, pooling, encoders, models, data, engine, synth
scripts/         prepare_mfa, extract_features, train, report
tests/           33 test (chạy được không cần GPU hay dữ liệu thật)
```