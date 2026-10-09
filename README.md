# Faces to numbers to predictions

Sort photos of two people into folders, measure 12 plain face numbers per photo with OpenCV, and sort new photos with logistic regression. No deep learning, no face recognition library. You can read every measurement in the notebook.

Start with the demo data so you see the full loop work. Then swap in your own photos.

## Try the demo first

```bash
pip install -r requirements.txt
python make_dataset.py
jupyter notebook face_classifier.ipynb
```

Then choose Run All in the notebook. `make_dataset.py` builds 40 drawings per person in `dataset/person_a/` and `dataset/person_b/`, plus two held-out drawings in `data/new_photos/` for the last step.

First run measures each photo and saves the table to `data/processed/features.csv`. That takes about a minute. Later runs reuse that file. Set `REBUILD = True` in the notebook when your photos change.

## Use your own photos

Only use photos of people who said yes, and who know how the photos will be stored and shared. Classmates often trade photo sets for this. Do not scrape friends, classmates, or strangers.

What tends to work:

* 20 to 40 frontal photos per person. More helps, but this setup stays small on purpose.
* Plain background, even daylight, face looking at camera, no sunglasses or heavy shadow.
* Some variety across days, rooms, shirts, and hair. If all 30 shots come from one burst, the test score looks better than it is.
* Crop is not needed. The notebook finds the face box itself.

Keep the layout exactly like this. Folder names become the labels:

```text
dataset/
  tushar/
    img001.jpg
    img002.jpg
    ...
  mudit/
    img001.jpg
    img002.jpg
    ...
```

Supported endings are `.jpg`, `.jpeg`, and `.png`. The code skips anything else, so convert `.bmp`, `.webp`, or `.heic` first.

Point the notebook at your folders:

```python
DATASET_ROOT = Path("dataset")
REBUILD = True
```

Run All again. Check the failure table in section 4. A photo gets skipped when the face, eyes, mouth, or brows cannot be found. That is normal on side views or dark shots. If more than a third get skipped, retake with better light and a straighter angle instead of tweaking thresholds.

Save one or two fresh photos per person outside `dataset/`, for example in `data/new_photos/`, and try them in section 10. Those photos train nothing. They just show what the fitted model thinks.

A note on file names and the train/test split: section 6 groups `img001` to `img008` into one session, `img009` to `img016` into the next, and keeps whole sessions in train or in test. This stops near duplicates from leaking across the split. If you keep that naming, you get that behavior for free. If you use your own names like `ana_kitchen_1.jpg`, replace the `session_of` line with something that fits, for example one folder per day, or one session per person if you only shot once.

## How to read your results

Section 5 plots two features so you can see if the classes pull apart. Section 8 prints train accuracy, test accuracy, and a confusion matrix.

Expect noise. With 40 to 70 usable photos, moving two or three test photos changes the score a lot. A gap like train 1.00 and test 0.85 usually means the features fit the training sessions too closely, or the test session had different light. Add another session per person before you trust the number.

The 12 features are ratios plus one iris color number: face shape, eye spacing and size, eye to mouth distance, mouth width and height, brow gap and thickness and angle, forehead and jaw width, and iris hue with a bit of saturation. The `image` column stays in the CSV for debugging and never enters the model.

## If it breaks

* `need at least 2 class folders`: `DATASET_ROOT` points at the wrong place, or one folder is empty.
* `no usable faces`: every photo was skipped. Run with `REBUILD = True` and read the reason column. Common reasons are `no face found` (crop too tight, profile view) and `eyes not found` (glare, glasses, dark).
* Stale numbers after swapping photos: delete `data/processed/features.csv` or set `REBUILD = True`.
* Haar files missing: `models/haarcascades/` lets detection run offline. If that folder is gone, the notebook falls back to the copies shipped with OpenCV.

## Files

* `face_classifier.ipynb`: the whole pipeline, read top to bottom.
* `make_dataset.py`: rebuilds the demo drawings.
* `dataset/`: class folders, demo at first, yours after.
* `data/processed/features.csv`: cached measurements.
* `data/new_photos/`: held-out photos for section 10.
* `models/haarcascades/`: local face and eye detectors.

## Limits

Demo accuracy says nothing about real faces. The brow, jaw, forehead, and iris steps are rough guesses for frontal, evenly lit shots. Anything hard is skipped and logged rather than guessed. This is a classroom tool for learning features and logistic regression, not for identifying people you do not know.
