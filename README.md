# COVID-19-XRAY-Detection

## Arsitektur Model

### Conv2D
Lapisan Convolutional berfungsi untuk mengaplikasikan filter ke gambar untuk mengambil karakteristik pada gambar

model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)))
- 32 = Jumlah filter yang digunakan
- kernel_size = ukuran dari piksel yang akan diaplikasikan filter
- activation = fungsi aktivasi
- input_shape = ukuran dari gambar yang masuk (diperkecil agar lebih cepat), (panjang, tinggi, jumlah channel warna) (3 berarti RGB)

- https://stats.stackexchange.com/questions/196646/what-is-the-significance-of-the-number-of-convolution-filters-in-a-convolutional
-https://www.geeksforgeeks.org/keras-conv2d-class/
-https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/

### MaxPooling2D
Lapisan Pooling berfungsi untuk mengurangi ukuran hasil lapisan Convolutional agar proses lebih cepat dan fitur dominan gambar bisa diperoleh

model.add(MaxPooling2D(pool_size = (2,2)))
- pool_size = ukuran dari piksel yang akan dipilih nilai terbesar

- https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

### Dropout
Lapisan Dropout berfungsi untuk menghilangkan input dari layer untuk mereduksi overfitting. Dropout = 0.25 berarti 25% dari jumlah neuron akan di set 0 atau tidak dipakai
 - https://www.oreilly.com/library/view/machine-learning-for/9781786469878/252b7560-e262-49c4-9c8f-5b78d2eec420.xhtml
 - https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5
 -https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/

### Flatten
Lapisan flatten berfungsi untuk merubah dimensi data. Contoh yang sebelumnya (2,2) menjadi (4)
- dropout_3 (Dropout)          (None, 12, 12, 128) 
    - 12, 12 = ukuran dari output layer dropout
    - 128 = jumlah filter yang ada
- flatten (Flatten)            (None, 18432)
- 18432 = 12 * 12 * 128

- https://stackoverflow.com/questions/44176982/how-does-the-flatten-layer-work-in-keras
- https://www.tutorialspoint.com/keras/keras_flatten_layers.htm


### Dense
Dense adalah neural network yang akan mempelajari gambar. Dense(64) berarti terdapat 64 neuron
.
- Diakhir model, nilai Dense adalah 1 atau hanya terdapat 1 neuron karena kita ingin mengetahui apakah terdapat COVID atau TIDAK, apakah neuron aktif atau tidak.
- https://stackoverflow.com/questions/43755293/what-does-dense-do

### Loss Function
adalah fungsi yang digunakan untuk mengestimasi tingkat error dari model agar neural network dapat belajar dan mengatur bobot dari neuron agar tingkat error dapat berkurang pada iterasi selanjutnya. Dalam hal ini fungsi yang digunakan adalah binary karena kelas klasifikasi adalah 2 yaitu COVID atau TIDAK.

- https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

### Optimizer
adalah fungsi yang digunakan untuk mengoptimasi performa neural network dengan merubah atribut jaringan seperti bobot dan learning rate
- https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html
- https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

---
### ImageDataGenerator
adalah fungsi yang digunakan untuk melakukan data augmentation. Hal ini dilakukan untuk membantu model menggeneralisasi data lebih baik dan mencegah overfitting
- rescale = merubah skala gambar rgb (0-255) menjadi (0,1) agar model lebih mudah untuk memproses data
- shear_range = men-skew gambar sebesar nilai yang dispesifikasikan
- zoom_range = men-zoom gambar sebesar nilai yang dispesifikasikan
- horizontal_flip = memutar gambar secara horizontal

- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
- https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/

### flow_from_directory
adalah fungsi yang digunakan untuk mengambil gambar dan mengidentifikasi kelas dari gambar tersebut berdasarkan nama folder. 
- target_size = ukuran dari gambar
- batch_size = jumlah gambar yang diambil setiap batchnya (berarti 32 gambar akan diambil dan dimasukkan ke model)
- class_mode = mode dari kelas (binary karena dua kelas yg diklasifikasi)

- https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

### fit_generator
- steps_per_epoch = jumlah batch yang dimasukkan ke model untuk dipelajari sebelum 1 epoch dianggap selesai
- epochs = nilai yang mendefinisikan berapa kali algoritma pembelajaran akan bekerja melalui seluruh dataset pelatihan 
- validation_steps = mirip seperti steps_per_epoch tetapi pada validation dataset

- https://datascience.stackexchange.com/questions/47405/what-to-set-in-steps-per-epoch-in-keras-fit-generator
- https://datascience.stackexchange.com/questions/29719/how-to-set-batch-size-steps-per-epoch-and-validation-steps
- https://stackoverflow.com/questions/44907377/what-is-epoch-in-keras-models-model-fit

---
### Useful Learning Resources

- https://youtu.be/aircAruvnKk
- https://youtu.be/wQ8BIBpya2k
- https://youtu.be/RfsF7_P5CW8
