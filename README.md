# SpareNTM (Sparsity Reinforced and Non-Mean-Field Topic Model)
SpareNTM is an open-source Python package implementing the algorithm proposed in the paper (Chen and Wang etc, ECML PKDD 2023), created by Jiayao Chen. For more details, please refer to [this paper](https://link.springer.com/chapter/10.1007/978-3-031-43421-1_9).

If you use this package, please cite the paper: Jiayao Chen, Rui Wang, Jueying He and Mark Junjie Li. Encouraging Sparsity in Neural Topic Modeling with Non-Mean-Field Inference. In Proceedings of ECML PKDD 2023, pp. 142-158.

If you have any questions or bug reports, please contact Jiayao Chen (chenjiayao2021@email.szu.edu.cn).

## 1. Requirements

- python==3.6
- tensorflow-gpu==1.13.1
- numpy
- gensim

## 2. Prepare data
Note: the data in the path ./data has been preprocessed with tokenization, filtering non-Latin characters, etc before, from [Scholar](https://github.com/dallascard/SCHOLAR) and the [Search Snipppets](http://jwebpro.sourceforge.net/data-web-snippets.tar.gz).

## 2. Run the model

  python SpareNTM.py --learning_rate 0.0001 --dir_prior 0.02 --bern_prior 0.05 --bs 200 --n_topic 50 --warm_up_period 100 --data_dir ./data/20ng/ --data_name 20ng
