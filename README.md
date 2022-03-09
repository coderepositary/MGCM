# MGCM
MGCM: Multi-Modal Generative Compatibility Modeling for Clothing Matching

Environment: python 3.5 tensorflow 1.2.1 cuda 8.0 cudnn 5

文章挖掘了GAN在单品兼容性建模中的潜力。GAN首次在图像生成领域被提出，旨在利用其生成器与判别器之间的对抗学习策略，生成尽可能真实的图像。事实上，其对抗学习的策略不仅在图像生成任务上获得了广泛成 功，也在表示学习任务上取得了优异表现。因此，GAN同时具有逼真的图像生成能力和强大的表示学习能力，不仅有利于对潜在兼容规律直接刻画，也有助于增强对时尚单品多模态数据的表示学习，因而具备对复杂兼容关系的全面理解和深度建模的潜力。 

鉴于此，本章通过引入GAN，基于给定单品（如上衣）的多模态数据，生成与之互补（如下衣）且兼容的潜在模板。通过该模板捕捉互补单品之间的兼容规律，并将其视为辅助关系桥梁，增强单品与单品之间的兼容关系建模，提高模型效果。 

本文设计的模型图如下，主代码见MGCM.PY: 
![QQ图片20220309164324](https://user-images.githubusercontent.com/43019981/157405657-713a5452-a0a6-4f08-a57b-e133c2921bdb.png)


在CGAN的基础上，我们设计了一个生成器,如下图所示，代码见generator.py：
![QQ图片20220309164943](https://user-images.githubusercontent.com/43019981/157406073-43eaba4b-e257-4bc0-a4b4-01e3cc38e09b.png)
