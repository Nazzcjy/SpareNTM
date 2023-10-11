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

## 3. Run the model
    python SpareNTM.py --learning_rate 0.0001 --dir_prior 0.02 --bern_prior 0.05 --bs 200 --n_topic 50 --warm_up_period 100 --data_dir ./data/20ng/ --data_name 20ng -output_dir ./output/

`--learning_rate`: Specify the learning rate.

`--dir_prior`: Specify the value of the Dirichlet prior.

`--bern_prior`: Specify the value of the Bernoulli prior.

`--bs`: Specify the number of the batch size.

`--n_topic`: Specify the number of topics. The default value is 50.

`--warm_up_period`: 

`--data_dir`: Specify the path to the input corpus file.

`--data_name`: Specify the name of the input corpus file.

`--output_dir`: Specify the path to the output directory.

## 4. Evaluation
topic coherence: [NPMI](https://github.com/jhlau/topic_interpretability).

topic diversity: TU.

topic quality: NPMI x TU

    bash run_computeQuality.sh

## Citation
If you want to use our code, please cite as

    @inproceedings{DBLP:conf/pkdd/ChenWHL23,
    author       = {Jiayao Chen and
                    Rui Wang and
                    Jueying He and
                    Mark Junjie Li},
    title        = {Encouraging Sparsity in Neural Topic Modeling with Non-Mean-Field
                    Inference},
    booktitle    = {Machine Learning and Knowledge Discovery in Databases: Research Track
                    - European Conference, {ECML} {PKDD} 2023, Turin, Italy, September
                    18-22, 2023, Proceedings, Part {IV}},
    series       = {Lecture Notes in Computer Science},
    volume       = {14172},
    pages        = {142--158},
    publisher    = {Springer},
    year         = {2023},
    url          = {https://doi.org/10.1007/978-3-031-43421-1\_9},
    doi          = {10.1007/978-3-031-43421-1\_9},
  }
