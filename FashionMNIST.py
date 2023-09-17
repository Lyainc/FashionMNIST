# 필수 라이브러리에 대한 import 부분입니다. 예시 코드와 차이점이 있다면
# 이미지 데이터는 다차원 배열이기 때문에 1차원 배열로 평탄화 해주는 Flatten을 추가로 import하였습니다.

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

#데이터셋을 불러와 각각 학습용 데이터와 테스트용 데이터로 나누었습니다.

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# FashionMNIST 데이터셋은 10가지의 클래스로 분류되므로, 이에 대한 레이블을 정의해주었습니다.

class_names = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터를 시각화해서 확인하는 코드입니다. google colab이나 Jupyter Notebook 사용시 해당 코드는 유의미하지만 
# 이번 과제 수행시 VSCode와 터미널을 사용하였으므로 해당 코드는 사용되지 않았습니다.

# 데이터셋의 이미지를 띄우고, 이를 250개의 컬러 코딩을 사용하여 확인할 수 있도록 합니다.

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# 2위와 동일하게 컬러코드를 사용해 맨 앞 데이터부터 25개를 5*5로 확인하는 코드입니다.
# 이미지 하단에는 지정되어 있는 라벨을 띄워줍니다.

plt.figure(figsize=(10, 10))
for i in range(25) :
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
# 기본 데이터의 픽셀값이 0~255까지 있으므로 데이터를 0~1 범위의 float값으로 표준화하기 학습용 데이터와 테스트용 데이터를 255.0으로 나누어줍니다.

train_images = train_images / 255.0
test_images = test_images / 255.0

# 이 모델은 순차구조 모델로 입력층부터 출력층을 차례대로 쌓아나갑니다. 

model = Sequential()

# 데이터가 다차원 배열이므로 이를 평탄화(Flatten)해서 1차원 배열로 만들어줍니다. input값은 이미지의 사이즈인 28*28로 지정했습니다.

model.add(Flatten(input_shape=(28,28)))

# 아웃풋을 128개로 하고 은닉층에 주로 사용하는 ReLu 활성화 함수를 사용합니다.  

model.add(Dense(128, activation='relu'))

# 최종적으로 10개의 클래스로 분류해아하므로 아웃풋을 10개로 하고, 다중 클래스 분류에 주로 사용되는 softmax 활성화 함수를 사용합니다.

model.add(Dense(10, activation='softmax'))

# 옵티마이저를 adam, 평가 지표는 accuracy를 사용합니다.
# 데이터가 일반 자료형이므로 손실함수를 sparse_categorical_crossentropy를 사용합니다.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 피팅을 시작합니다. 64 배치 사이즈로 묶어 10번을 반복합니다. 모델의 정확도를 검증하기 위해 학습데이터 셋에서 validation set를 별도로 둡니다.
# verbose는 출력의 형태를 의미하며, 자세하게 출력하기 위해 1의 값을 넣어줍니다.

model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2, verbose=2)

# 모델을 테스트합니다.

test_loss, test_acc = model.evaluate(test_images, test_labels)

# 테스트 정확도를 프린트합니다.

print('Test accuracy:', test_acc)

# 학습된 모델로 실제 테스트 데이터를 분류합니다.

predictions = model.predict(test_images)

# 테스트 데이터의 첫번째 이미지를 분류하고, 각 클래스에 해당될 확률을 배열로 출력합니다. (0~9)

print(predictions[0])

# 가장 높은 확률의 클래스 인덱스를 출력합니다.

print(np.argmax(predictions[0]))