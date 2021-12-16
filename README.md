# Features_decomposition
Learning to decompose features into class-specific and domain-specific Representations.



For architecture search
Run it using python arch_search.py --data_source 'path/to/Data/Original_images/amazon/images' --data_target 'path/to/Data/Original_images/webcam/images' --source_domain 'amazon/webcam/dslr' --target_domain 'amazon/webcam/dslr'
Copy the genotypes of the searched cell to genotypes.py 'PCDARTS' and run 
python train_fedec.py --arch PCDARTS --data_source 'path/to/Data/Original_images/amazon/images' --data_target 'path/to/Data/Original_images/webcam/images' --source_domain 'amazon/webcam/dslr' --target_domain 'amazon/webcam/dslr'
