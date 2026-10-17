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
| Wrapper WavLM Base+ (`paid/encoders.py`)                 | <Đã chạy / Chưa chạy>                                             |
| MFA trên L2-ARCTIC thật                                  | Đã chạy, căn chỉnh được <N_ok>/<N_total> câu (bỏ qua <N_skip>)    |
| Con số 81.41% của bài báo                                | Bản dựng này đạt <xx.xx> ± <x.xx>% (xem "Kết quả" bên dưới)       |

Môi trường chạy: <GPU>, Python <x.y>, torch <x.y.z>, transformers <x.y.z>, MFA <3.x>,
commit `<hash>`, ngày <dd/mm/yyyy>.

## Kết quả

Whisper-small, 4-fold speaker-disjoint, 10 epoch, mean ± std theo fold (ddof=0):

| Biến thể         | Accuracy (%)  | Ghi chú                          |
| ---------------- | ------------- | -------------------------------- |
| `utt`            | <xx.xx ± x.xx> |                                  |
| `paid_nopid`     | <xx.xx ± x.xx> |                                  |
| `paid`           | <xx.xx ± x.xx> | Bài báo: 81.41                   |
| `speechutt_proj` | <xx.xx ± x.xx> | Control chính (xem phần đọc kết quả) |

<Một hai câu nhận định thật: có tái hiện được không, chênh bao nhiêu, và `paid` so với
`speechutt_proj` cho thấy điều gì. Nếu có fold yếu bất thường (ví dụ fold 3) thì nêu ở đây.>

Số từng fold không khớp bài vì cách chia speaker vào fold khác (xem bảng quyết định bên dưới);
chỉ nên so trung bình.

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
Lặp lại với `feats/wavlm` (thêm `paid_nopid`, `frameattn`) để dựng lại Bảng I.

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
