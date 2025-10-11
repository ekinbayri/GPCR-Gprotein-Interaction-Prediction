This project is a masters thesis in collaboration with Bio2Byte. 

The project implements transfer learning on a deep learning model for protein-protein interaction (PPI) prediction. 

The intermediate phase of the model is removed and transfer learning is implemented using pan dataset (PPI dataset) as source domain and two human GPCR-G-Protein interaction datasets (chimeric G-Proteins and natural G-Proteins) are used for target domains. 

Positive transfer was achieved on natural G-Proteins but not in chimeric G-Proteins. 


The requirements for running the deep learning model xCAPT5 can be found on: https://github.com/aidantee/xCAPT5/

The Chimeric_GPCR_Dataset, GPCR_Dataset, and Pan_Dataset folders contains the datasets used in the project. 

The residue embeddings required can be generated using generate_embedding.py file in the Utility folder.

The Training folder contains the xCAPT5 model and implemented changes. 

The file can be run in 3 modes; training, testing, and transfer learning.

Training mode is used to train the model on source domain before transfer learning on target domain.

Testing can be used to interpret results with plots and metrics.

Transfer learning mode implements transfer learning on target domain.
