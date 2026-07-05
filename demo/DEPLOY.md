# DALM-DETR demo — lokal sinov va HF Spaces'ga deploy qilish

## 0. Fayllarni joylashtirish (repo ichida)

```bash
mkdir -p demo/examples
# app.py, requirements.txt, DEPLOY.md fayllarini demo/ ichiga ko'chiring
```

## 1. Lokal sinov (deploy'dan OLDIN, majburiy)

```bash
pip install gradio
python demo/app.py
```

Brauzerda http://127.0.0.1:7860 ochiladi. Hozircha faqat `baseline_v3`
checkpoint bor — `ours_adaptive_v3` tayyor bo'lguncha sinov uchun ikkala
slotga ham baseline'ni qo'yib turish mumkin:

```bash
export DALM_ADAPTIVE_FILE=offsetiou_baseline_v3_best.pth
python demo/app.py
```

Agar `build_model()` xatolik bersa (konstruktor nomi mos kelmasa), shu
buyruq natijasini Claude'ga yuboring — bir qatorda tuzatamiz:

```bash
grep -n "^def \|^class " src/offsetiou_det/models/detector.py
```

## 2. Checkpointlarni HF Hub'ga yuklash (training tugagach)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # tokeningizni kiriting
huggingface-cli repo create dalm-detr --type model
huggingface-cli upload SIZNING_USERNAME/dalm-detr \
    outputs/checkpoints/offsetiou_baseline_v3_best.pth \
    offsetiou_baseline_v3_best.pth
huggingface-cli upload SIZNING_USERNAME/dalm-detr \
    outputs/checkpoints/offsetiou_ours_adaptive_v3_best.pth \
    offsetiou_ours_adaptive_v3_best.pth
```

## 3. Space yaratish

1. huggingface.co → New Space → SDK: **Gradio**, Hardware: **CPU basic (free)**
2. Space repo'siga quyidagilarni yuklang:

```
app.py                      (demo/app.py — ILDIZGA, nomi aynan app.py)
requirements.txt            (demo/requirements.txt)
src/offsetiou_det/          (butun paket papkasi)
examples/                   (3-4 ta gavjum sahna .jpg — CrowdHuman'dan EMAS,
                             litsenziya CC BY-NC; o'zingiz olgan yoki
                             Unsplash/Pexels'dan bepul rasmlar)
```

3. Space → Settings → Variables:

```
DALM_HF_REPO = SIZNING_USERNAME/dalm-detr
```

## 4. README badge

README.md tepasiga (train tugagach):

```markdown
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Spaces-blue)](https://huggingface.co/spaces/SIZNING_USERNAME/dalm-detr)
```

## Eslatmalar

- app.py ichidagi GitHub linkida `YOUR_USERNAME` ni o'zingiznikiga almashtiring.
- CrowdHuman rasmlarini Space'ga yuklamang (CC BY-NC — qayta tarqatish yo'q).
- CPU'da bitta rasm ~2-5 s; ikkita model ketma-ket ishlaydi, bu normal.
