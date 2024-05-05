# Plant-Identification-and-Disease-Detection
Convolutional Neural Networks (CNNs), Image Segmentation,Feature Extraction,Pattern Recognition, Machine Learning,Deep Learning


Plant Identification and Disease Detection
1 KEERTHANA.P,2CHALUKYA NAYAKA B K,3 K B BHARATH,4HARSHAVARDHAN C,5 PUNITH V 
1 Assistant Professor, Computer Science and Engineering ,Reva University
2,3,4,5 Students, Computer Science and Engineering ,Reva University




Abstract- Many Machine Learning (ML) models have been employed for the detection and classification of plant diseases but,Early detection of plant diseases is crucial to ensure healthy crop growth. Deep learning (DL), a powerful subset of machine learning, offers significant advantages in this area by achieving high accuracy in disease classification. This review explores the application of various DL architectures and visualization techniques for identifying plant disease symptoms from images. We discuss the evaluation metrics used to assess these methods and highlight the potential for even earlier disease detection before visible symptoms appear, addressing current research gaps.

Keywords- Convolutional Neural Networks (CNNs),
Image Segmentation,Feature Extraction,Pattern Recognition,
Machine Learning,Deep Learning

INTRODUCTION

Effective detection of plant diseases is essential for maintaining robust crop growth and minimizing yield losses. While traditional machine learning models have been used for disease classification, deep learning (DL) represents a powerful advancement. DL excels in disease identification due to its ability to extract intricate patterns from image data. In this review, we delve into various DL architectures and visualization techniques applied to pinpoint plant disease symptoms within images. We explore evaluation metrics and discuss the exciting potential of detecting diseases even before visible symptoms manifest, addressing critical gaps in current research.Plant diseases significantly impact global agricultural productivity, making timely detection crucial for effective crop management. Traditional methods for disease identification are labor-intensive and require specialized expertise. However, recent breakthroughs in deep learning have revolutionized digital image processing, surpassing traditional approaches. Researchers are now exploring how to leverage deep learning technology for plant disease identification, combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to enhance accuracy. These advancements offer promising solutions to the challenges associated with manual disease detection and pave the way for more efficient and accurate methods.


II.  RESEARCH GAP

The examination for the most part founded on the plant leaf structure,size,shape and variety 87 investigations utilized leaf shape and 13 investigations utilized blossom shape for plant species ID. The surface of leaves and blossoms is broken down by 24 and 5 examinations individually. Variety is chiefly considered alongside leaf size investigation (9 examinations), however a couple of concentrates likewise involved variety for blossom examination (5 studies).From the exploration the leaf spot are the normal diseases influencing plants .
Most organization models are worked on in view of the current fantastic organizations, and the acknowledgment precision is additionally getting to the next level. the effect of organization model boundaries and preparing time on the acknowledgment productivity. In the event that these models are relocated to the versatile terminal later on, the model acknowledgment productivity will be low, can't meet the day to day needs, and can't be placed into genuine creation. The organization model distinguishes 10 illness pictures of five kinds of harvests and joins movement learning, he paper directs out that the current strategies toward work on the presentation of CNN depend on expanding profundity, width, and goal. Nonetheless, if by some stroke of good luck one boundary is upgraded, it will prompt issues like inclination misfortune, slope blast, a lot of computation

     III  LITERATURE SURVEY

Apple plantations are frequently tormented by infections, for example, Alternaria leaf spot, Earthy colored spot, Mosaic, Dim spot, and Rust; these five illnesses are infamous for their extreme effect on apple creation. Regardless of this, there is a remarkable hole in the exploration for an identification framework that is both exact and convenient, which is fundamental for the solid movement of the apple business. Object location systems like SSD, DSSD, and R-SSD are commonly separated into two portions: the pre-network model which goes about as an essential element extractor, and a helper structure that influences multi-scale highlight maps for upgraded identification capacities.

The K-implies calculation divides the leaf image into four bunches using squared Euclidean distances as the distance metric. To capture both variety and surface highlights, include extraction is carried out using the Variety Co-event method. In this method, characterization is achieved by using a brain network position computation considering the backpropagation system. The framework's overall accuracy for infection detection and arranging tasks is approximately 93%.

A few harvest types, including organic product, vegetable, oat, and business crops, are dependent upon contagious illness identification on their leaves. Unmistakable strategies are utilized for every class.
On account of natural product crops, division is directed utilizing k-implies grouping, with an emphasis on surface highlights characterized through ANN and closest neighbor calculations. This approach yields a noteworthy typical precision of 90.723%.
For vegetable yields, division depends on the Chan-Jar strategy, while surface component extraction uses nearby twofold examples. Characterization is achieved through SVM and k-closest neighbor calculations, bringing about a general normal precision of 87.825%.

IV. METHODS OF DISEASE DETECTION
     
The plant disease detection system operates in four main stages. Initially, the system captures images using a digital camera, mobile phone, or sources them from the internet. Following this, the second stage involves segmenting the image into multiple clusters, employing various techniques for this purpose. The subsequent stage is dedicated to extracting features from the image. Finally, the process concludes with the classification of the diseases detected in the plants as shown in fig 3.1

                   Fig: 4.1 Phases of plant disease detection system
     Image Acquisition
With the help of electronic devices like cameras or smartphones, leaf photos can be taken at the perfect angle and size. Additionally, images can be retrieved from web archives.
The application framework designer has complete control over the creation of an image database. The image data set plays a crucial role in improving the classifier's productivity in the last stage of the illness identification framework.

Image Segmentation
With the help of sophisticated devices like cameras or cellphones, leaf images can be taken at the perfect angle and size. Electronic stores can also provide photographs.
The application framework engineer is solely responsible for creating an image data collection. The picture data set assumes a critical part in improving the proficiency of the classifier in the last period of the illness location framework.

Feature Extraction
In this step, the features from the area of interest need to be extracted. These features are necessary to determine the meaning of a sample image. Features can be based on color, shape, and texture. Recently, most researchers have focused on utilizing texture features for plant disease detection. There are various feature extraction methods that can be employed for developing the system, such as the gray-level co-occurrence matrix (GLCM), color co-occurrence method, spatial gray-level dependence matrix, and histogram-based feature extraction. The GLCM method is a statistical technique for texture classification.

Classification
During the classification phase, the objective is to determine the health status of the input image, discerning if it’s indicative of a disease. In cases where the image is diagnosed with a disease, various techniques are then applied to further classify the type of disease present. To accomplish this, a software routine is crafted, typically in MATLAB, serving as the classifying tool.

              Fig: 4.2 METHOD OF DISEASE DETECTION

In the past few years, a range of classifiers such as k-nearest neighbor (KNN), support vector machines (SVM), artificial neural networks (ANN), backpropagation neural networks (BPNN), Naïve Bayes, and decision tree classifiers have been implemented by researchers. SVM, in particular, has become the preferred choice due to its widespread use. While each classifier has its unique advantages and limitations, SVM is noted for being straightforward and reliable

V. OVERVIEW OF PLANT DISEASE

Plant diseases are frequently caused by enticing experts such as infections, growths, and microscopic creatures. Plant disease symptoms provide the observable evidence of contamination, whereas side effects are the obvious consequences of these diseases. Signs of contagious contaminations include obvious spores, accumulation, or form; typical side effects include yellowing and patches on leaves. Plant contaminations that spread quickly are caused by parasites, which can be either single-celled or multicellular organisms that contaminate plants by severing tissue and consuming nutrients.These represent the most prevalent categories of plant diseases. Fungal diseases in plants can cause spots on leaves, yellowing of the foliage, and bird's eye spots on berries, among other obvious signs. The fungal organism itself may occasionally be seen clearly as a growth or mold on the leaves.


                Fig 5.1 Leaf affected by fungal infection

Abnormal growths can surface on the stems or the bottom side of leaves, serving as direct indicators of a pathogen’s presence, known as infection signs. Bacteria are single-celled organisms without a distinct nucleus and are found nearly everywhere. Although numerous bacteria are harmless or beneficial, others can be harmful, causing diseases in humans and plants alike. Spotting signs of bacterial infection is typically more difficult than spotting fungal infections due to the minuscule size of bacteria. Cutting into an infected stem might reveal a white, creamy secretion known as bacterial ooze, a telltale sign of bacterial infection. Other symptoms include lesions that are drenched, appearing as moist blotches on leaves that exude bacteria. With the disease’s progression, these lesions grow, turning into reddish-brown stains on the leaves. A common sign of bacterial infection is the appearance of spots on leaves or fruits, which are usually restricted by the veins of the leaf, distinguishing them from the more scattered fungal spots.


                      Fig 5.1 Leaf affected by bacteria
The particles that make up an infection are too little for a light magnifying lens to be able to distinguish them. They assault the host's cells and seize its equipment in order to force the host to produce a significant number of copies of the infection. Plants that are infected with viruses do not exhibit symptoms since the diseases themselves should be invisible, even under a strong light microscope. Nevertheless, the trained eye can detect certain negative effects. Viral contamination is typically accompanied with crinkled or yellowed leaves with a mosaic-like pattern. Many plant illnesses, such as tobacco mosaic infection, acquire their names from this excellent example of staining. Similarly, plant development is also typically reduced in viral infections.


                                Fig 5.3 Leaf affected by virus

Hence, we share our insights on identifying various plant diseases and the measures required for their management.

The suggested framework includes a TFLite-powered Android application from start to finish. The suggested framework was chosen to support the development of an Android app that can identify plant diseases. It uses Convolutional Brain Organization in its computations and models to identify diseases and species in the yield leaves. The suggested framework modifies source code using Colab. Plant Town is a collection of 54,305 images of firm and ill plant leaves that were taken under controlled circumstances. Apple, blueberry, cherry, grape, orange, peach, pepper, potato, raspberry, soy, squash, strawberry, and tomato are among the fourteen harvest varieties seen in the images. It has images of seventeen major diseases, four bacterial diseases, two shape-related (oomycete) infections, two viral diseases, and one disease caused by a rodent.Additionally, 12 yield species have solid leaf images that don't seem to be affected by illness. We have responses in our dataset for a few different plant surfaces,


such as,


Our framework is outfitted with information generators that filter through pictures in determined envelopes, convert them into 'float32' tensors, and convey them to our brain organization, complete with their names.

Brain networks perform best with standardized information.  Thusly, we will align our picture pixels from their unique [0, 255] territory to a standardized [0, 1] territory. Furthermore, we will change the size of the info pictures to meet the organization's details, either 224x224 or 299x299 pixels. Whether to apply picture increase ultimately depends on you.

In addition, our framework doesn't simply identify plant sicknesses; it likewise associates clients to a web-based commercial center. This entrance shows a variety of pesticides designated at the analyzed sickness, each with its MRP recorded. The site likewise gives utilization bearings. Subsequently, clients can pursue informed choices while buying the most appropriate pesticide in the wake of contrasting costs and highlights.

  VI. Results and discussion

 The way the plant infection finding model is presented can change significantly depending on the environment in which it is tested. Certifiable outdoor conditions provide challenges because of differences in lighting, foundation, and picture qualities, whereas lab conditions—where the model is built and tested on a similar dataset—may produce very accurate results. When the model is applied to field photos, the difference between the lab and handle settings can cause a large decrease in its accuracy. We suggested displaying heterogeneity in the preparation dataset as a solution to this problem by combining a variety of photo blends that included images taken in various real-world scenarios.We hope to close the accuracy gap between lab and real-world settings by strengthening the model's robustness and field performance during training by exposing it to a larger range of image circumstances.


               Fig 6.1 Visualization of Accuracy Result

  The proposed framework beats the constant apple leaf illness identification that utilizes a profound learning procedure with cutting edge convolutional brain organizations, chiefly because of its capacity to recognize numerous sicknesses in a single framework at the same time.





   VII.CONCLUSION

   The conclusion indicates the successful enhancement of an application that focuses on accuracy in real-world situations to distinguish between ill and healthy plants. The program was done without any preparation, providing acceptable precision, using a variety of plant sickness photographs. In order to improve exactness even more, future efforts will concentrate on expanding the image data collection and improving the architecture. This project suggests a significant advancement in plant illness location innovation and offers beneficial benefits for agricultural partners. The focus on true accuracy draws attention to its potential impact on enhancing harvest production and well-being. As developments continue, ongoing improvement will be essential to effectively handle growing agricultural challenges. Together with acceptable agricultural practices around the plant disease site, cooperative initiatives will propel more advancements in the field.

   VIII   REFERENCES

1.P. K. Sethy, N. K. Barpanda, A. K. Rath and S. K. Behera, "Deep feature based Rice leaf disease identification using support vector machine", Comput. Electron. Agricult., vol. 175, Aug. 2020.

2. S. Sankaran, A. Mishra, R. Ehsani and C. Davis, "A review of advanced techniques for detecting plant diseases", Computers and Electronics in Agriculture, vol. 72, no. 1, pp. 1-13, 2010.


3. H. Wang, G. Li, Z. Ma and X. Li, "Application of Neural Networks to Image Recognition of Plant Diseases", International Conference on Systems and Informatics(ICSAI2012), pp. 2159-2164, 19-20 May 2012.


4. M. Islam, K. Wahid AnhDinh and P. Bhowmik, "Detection of Potato Diseases Using Image Segmentation and Multiclass Support Vector Machine", IEEE 30th Canadian Conference on Electrical and Computer Engineering (CCECE), 30 April- 3 May 2017.


5. U. Mokhtar, M.A.S. Ali, A.E. Hassenian and H. Hefny, "Tomato leaves disease detection approach based on support vector machines", 11th international computer engineering conference (ICENCO), pp. 246-250, 29-30 Dec.2015.


6. W. Hongzhi and D. Ying, "An Improved Image Segmentation Algorithm Based on Otsu Method", International Symposium on Photo electronic Detection and Imaging SPIE, vol. 6625, 2008

7.Radical sound valuation of fetal weight with the use of deep learning"for publication in Proceedings of the 1st International Conference on Engineering, Medicine, Management, Arts and Sciences Volume 2742, Issue 1, AIP Conf. Proc. 2742, 020044 (2024) https://doi.org/10.1063/5.0184155

8. “Visualization of Data Structure and Algorithm”, International Journal for Research in Applied Science and Engineering Technology (IJRASET), Volume: 11, Issue: 04, April-2023, https://doi.org/10.22214/ijraset.2023.51094

9. “An Economical Deep Learning Framework for COVID-19 Diagnosis Using Lung Ultrasound
Images”, International Conference on Advances in Science, Engineering and Technology(ICASET-2022), ISBN : 9788770227896,doi: https://doi.org/10.13052/rp-9788770227896

