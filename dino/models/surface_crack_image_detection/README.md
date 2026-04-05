---
license: apache-2.0
metrics:
- accuracy
- f1
base_model:
- google/vit-base-patch16-224-in21k
---
Check whether there is a surface crack given surface image.

See https://www.kaggle.com/code/dima806/surface-crack-image-detection-vit for more details.

```
Classification report:

              precision    recall  f1-score   support

    Positive     0.9988    0.9995    0.9991      4000
    Negative     0.9995    0.9988    0.9991      4000

    accuracy                         0.9991      8000
   macro avg     0.9991    0.9991    0.9991      8000
weighted avg     0.9991    0.9991    0.9991      8000
```