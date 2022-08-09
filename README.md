# CS-BTM

This is a python implementation according to [CS-BTM: a semantics-based hot topic detection method for social network](https://doi.org/10.1007/s10489-022-03500-9).

The code structure is based on [a python BTM implement](https://github.com/MrNiro/BTM).

### Prerequisite
- [Rust](https://www.rust-lang.org/tools/install)
- Transformers
  
    ```pip install transformers==4.18.0 --ignore-installed PyYAML```
- TensorFlow：
  - CPU: ```pip install tensorflow==2.3.0``` 
  - GPU: ```pip install tensorflow-gpu==2.3.0```
- [pytorch](https://pytorch.org/get-started/locally/)

### How to use
- Run with default config and test data
    - Topic Learning, Topic Inference and Model Evaluation will be performed.    
    ```python src/evaluate.py```

- Edit configs in ```src/evaluate.py```

### What is CS-BTM
To consider the relationship between different two bitems, such as context semantics, polysemy, and similarity, incorporate Bert with BTM
