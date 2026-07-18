# lotti-llm

Sıfırdan eğitilen, küçük (~30-40M parametre, ~3-4k vocab) bir transformer ile:
1. **Pretraining** — Lottie (Bodymovin) JSON animasyon formatının yapısını/grammar'ını öğrenmek.
2. **Finetuning** — doğal dil açıklamalarından Lottie animasyonu üretmek (text-to-animation).

Eğitim [Google Colab](https://colab.research.google.com/) (Pro/Pro+, A100/V100) üzerinde çalışır. Kod bu repoda yazılır/düzenlenir; Colab yalnızca çalıştırma ortamıdır (bkz. "Colab'da çalıştırma" altında).

## ⚠️ Veri lisansı

Varsayılan pretraining/finetuning veri kaynağı [`OmniLottie/MMLottie-2M`](https://huggingface.co/datasets/OmniLottie/MMLottie-2M) **CC-BY-NC-SA-4.0** lisanslıdır:

- **NonCommercial**: bu veriyle eğitilen ağırlıklar türev eser sayılır ve ayrı bir ticari lisans olmadan ticari bir üründe kullanılamaz.
- **ShareAlike**: veri setinin veya ondan "esaslı şekilde türetilmiş" artefaktların yeniden dağıtımı aynı lisans altında olmalıdır.

Bu proje şu an araştırma/kişisel kullanım kapsamındadır. Ticari kullanım gündeme gelirse veri kaynağının değiştirilmesi veya ayrı bir lisans alınması gerekir.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Repo yapısı

- `src/lottiegpt/data/` — HF dataset indirme, filtreleme, train/val split.
- `src/lottiegpt/tokenizer/` — Lottie JSON <-> token dizisi (yapısal + kuantalanmış sayısal token'lar), metin alt-tokenizer'ı.
- `src/lottiegpt/model/` — sıfırdan yazılmış GPT-tarzı decoder-only transformer (RoPE, tied embedding).
- `src/lottiegpt/training/` — pretraining/finetuning döngüleri, checkpoint/resume, eval.
- `src/lottiegpt/inference/` — üretim + Lottie JSON doğrulama.
- `notebooks/` — Colab'da çalıştırılan ince orkestrasyon notebook'ları (gerçek mantık `src/` içinde).
- `scripts/` — tek seferlik doğrulama/analiz script'leri.

Detaylı tasarım ve milestone planı için proje geçmişindeki plan dosyasına bakılabilir.

## Colab'da çalıştırma

İlk hücre:

```python
!git clone https://github.com/<kullanici>/lotti-llm.git
%cd lotti-llm
!pip install -e .
```

Sonraki hücreler `!git pull` ile güncellenir, ardından ilgili modül çalıştırılır, ör:

```python
!python -m lottiegpt.training.pretrain --config configs/pretrain.yaml
```

Checkpoint ve büyük veri artefaktları Google Drive'da tutulur (`google.colab.drive.mount(...)`); kod her zaman GitHub'dan gelir.
