# Traffic Signs Detection![image](https://user-images.githubusercontent.com/45328601/149921779-2cf0b294-e36d-422f-9b42-84144d8b6c9a.png)

<img src="https://ai.github.io/size-limit/logo.svg" align="right"
     alt="Size Limit logo by Anton Lovchikov" width="120" height="178">

Araçların 100 metre ötedeki trofik işaretlerini yapay zeka teknolojisi ile kolay bir şekilde saniyelr için de 98% - 100% oranın da
doğru sonuçlar ile sürücüye bildirecek bir derin öğrenim projesidir.
Bu projenin yapımın da başta Numpy olmak la birlikte bir çok kütüphane den yararlanıldı (Kullanılan Kütüphanelere aşşağıda ki Lİsteden bulabilirisiniz), 
projenin başında Devasa denecek düzeyde bir veri kümesin den yararlandık yaklaşık 50 000 adet trafik işareti resminden yararlandık bu veri kümesi ne kadar 
artarsa projenin çıktı doğruluğu ve hızı daha çok artmaktadır. Kullanılan veri kümelerin de ki öğelerin piksel piksel incelenmesi için kares yapay zeka 
framework lerinden yararlandık bu saye de numpy framework' unun  bizler için derlediği veri kümesi kares tarafından teke rteker incelendi ve bilgisayarın 
işlemcisin de depolamıştı. Ardından bu veri kümelrini görüntüye işleme kısmı geldiğin de de bunlar için Open CV framework' ünü kullandık, bu bizim için cihazın
kalmerasından aldığı verileri base de bulunan verilere iletmektedir yani client ile sisitem arasında ki bağı sağlıyor. En son aşama olan bu işlemlerin hepsini
hakim bir şekilde yönetmemizi sağlayan çok kullanışlı ve popiler olan bir framework olan TensorFlow Dan yararlanıldı, bu aracı projemizin omurgası olarak düşüne 
biliriz çünkü tüm işlemleri bize işleyip sunacak olan TensorFlow dur. Aşşağıda projenin nasıl kullanılacağını Stap Stap görebilirsiniz.

* Numpy
* Matplotlib
* Keras
* Open CV
* Pickle
* OS
* Random
* TensorFlow

Cihaz kamerasından aldığı verileri drekt olarak işleyip çıktı sunmakta ve istatistiki verilerini de vermekte dir.

<p align="center">
<img src="https://github.com/yusufaras104/Open_CV/blob/main/assest/images/opencv.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="686" height="289">
</p>



## Nasıl Çalışıyor

1. Başlıca olarak sistemin çalışması için gereksinimleriini  sağlanması gerekmektedir, bunlar kurulumunun  yapılması gerekilen araçlardır; `Numpy`,
`Matplotlib`, `Keras`, `Open CV`, `OS`, `Pandas`, `Random`, `Random`, `TensorFlow`.
2. Ardından Git dosyamızda da ulaşabileceğimiz Kaynak kodlarını kullanarak eksik olan araçlarınızı takip edip gereksinimlerinizi tamamlaya bilir ve 
projeyi çalıştırabilirsiniz. [See here](https://github.com/yusufaras104/Open_CV/main.py)
3. Projenin çalıştırılmasının ardından cihazınız ilk önce veri kümesin de bulunan elemanları tanıyıp analiz etmesi gerekmektedir.
4. Tarama işleminin ardından yapay zeka araçlarımız bu verileri base de depolar ve istatistiksel olarak tanımlarını yapar.
5. taramalar ve öğrenimler yapıldık tan sonra görüntü işlememizi sağlayabileceğimiz aracımız olan Open CV devreye girer ve cihazın kamerasından aldığı 
verileri işlemci tarafından işlenmesi için bir sonra ki adım olan TensorFlow tarafına gönderilir.
6. TensorFlow Aracının aşamasına gelindiği zaman son noktaya ulaşılmış oluyoruz burada yukarıda yapılan tüm aşamalar teker teker gerçekleşmektedir ve bundan 
sonra sistem tam anlamıyla çıktı verip clinet tarafını input edilen çıktıya göre bilgilendirmiş olur. 

### Gerekli eklentilerin kurulumu

Burada pip dosya yöneticisi ile yararlanılmaktadır eğer pip yüklü değilse yüklenmesi gereklidir (```$ python -m ensurepip --upgrade```, <br> 
```$ python get-pip.py```, ```$ python -m pip install --upgrade pip ```)

<details><summary><b>Step Step Kurulum</b></summary>

1. Numpy :
``` pip install numpy ``` 
2. Matplotlib : ``` pip install matplotlib```
3. Keras : ```$ pip install --upgrade pip```, ```$ pip install tensorflow```, ```$ pip install tf-nightly```
4. Open CV :
  ```diff
  + git clone https://github.com/opencv/opencv
  + git -C opencv checkout <some-tag>
  + # optionally
  + git clone https://github.com/opencv/opencv_contrib
  + git -C opencv_contrib checkout <same-tag-as-opencv> 
  + # optionally
  + git clone https://github.com/opencv/opencv_extra
  + git -C opencv_extra checkout <same-tag-as-opencv>
  ```
5. Pandas:
  ```diff
1. İşletim sisteminiz için Anaconda'yı ve en son Python sürümünü indirin , yükleyiciyi çalıştırın ve adımları izleyin. Lütfen aklınızda bulundurun:

  + Anaconda'yı kök veya yönetici olarak yüklemek gerekli değildir (ve önerilmez).
  + Anaconda3'ü başlatmak isteyip istemediğiniz sorulduğunda, evet yanıtını verin.
  + Kurulumu tamamladıktan sonra terminali yeniden başlatın.
  + Anaconda'nın nasıl kurulacağına ilişkin ayrıntılı talimatlar Anaconda belgelerinde bulunabilir .

2. Anaconda isteminde (veya Linux veya MacOS'ta terminal), JupyterLab'ı başlatın:
  
  ```
  <img src="https://pandas.pydata.org/static/img/install/anaconda_prompt.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="686" height="289">
</p>
  ```
```diff
  3. JupyterLab'da yeni bir (Python 3) not defteri oluşturun:
  ,,,
  <img src="https://pandas.pydata.org/static/img/install/jupyterlab_home.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="686" height="289">
  
  ```diff
  4. Not defterinin ilk hücresinde pandaları içe aktarabilir ve sürümü şu şekilde kontrol edebilirsiniz:
  ```
  <img src="https://pandas.pydata.org/static/img/install/pandas_import_and_version.png"
  alt="Size Limit comment in pull request about bundle size changes"
  width="686" height="289">
  
   ```diff
  5. Artık pandaları kullanmaya hazırsınız ve sonraki hücrelere kodunuzu yazabilirsiniz.
  ```


</details>


