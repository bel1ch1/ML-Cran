# ML-Cran

Проект машинного обучения - для мостового крана.
Проект направлен на уменьшение раскаченность груза



Data structure
```py
[
  [timestamp1, timestamp2, timestamp3, timestamp4, ...],
  [X1, X2, X3, X4, X5, X6, X7, ...],
  [Y1, Y2, Y3, Y4, Y5, Y6, Y7, ...]
]
```
Description
- timestamp - временная метка
- x - отклонение от центра кадра
- y - отклонение от центра кадра
- mark - метка
---

Датасет не является настоящим. Был сгенерирован с использованием стороннего софта

- dataset 1000 rows
- batch = 20 rows
- Training data = 35 batchs
- Testing data = 15 batchs

10к epoch

![alt text](/REadme_files/image.png)

---

### RNN (LSTM) with correct data emitation

1k epoch, 1k rows

![altuha](/REadme_files/{6E63C1F1-A3A9-4A6E-9645-97A63AA5480B}.png)


### RNN with right data

10 epoch, 1k rows

![alt text]({4F164360-373E-483D-9B22-F7939AE27A50}.png)

100 epoch, 1k rows

![alt text]({FBF2B88B-6F7A-475E-90A4-EFF122FDF7C6}.png)


1k epoch, 1k rows -overfitting

![alt text]({40F7C2D0-8FA7-4C08-9796-7ED3AEFFFD4C}.png)
