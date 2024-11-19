<div align="center">

# GLAFF: Robust Time Series Forecasting

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2409.18696&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2409.18696)
![Visits Badge](https://badges.pufler.dev/visits/ForestsKing/GLAFF)
![Stars](https://img.shields.io/github/stars/ForestsKing/GLAFF)
![Forks](https://img.shields.io/github/forks/ForestsKing/GLAFF)

</div>

**Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective**

Time series forecasting has played a pivotal role across various industries, including finance, transportation, energy, healthcare, and climate. Due to the abundant seasonal information they contain, timestamps possess the potential to offer robust global guidance for forecasting techniques. However, existing works primarily focus on local observations, with timestamps being treated merely as an optional supplement that remains underutilized. When data gathered from the real world is polluted, the absence of global information will damage the robust prediction capability of these algorithms. To address these problems, we propose a novel framework named GLAFF. Within this framework, the timestamps are modeled individually to capture the global dependencies. Working as a plugin, GLAFF adaptively adjusts the combined weights for global and local information, enabling seamless collaboration with any time series forecasting backbone. Extensive experiments conducted on nine real-world datasets demonstrate that GLAFF significantly enhances the average performance of widely used mainstream forecasting models by 12.5%, surpassing the previous state-of-the-art method by 5.5%.

![](./img/architecture.png)

## 	Get Started

1. Install Python 3.10.13 and PyTorch 2.1.2.

2. Download datasets and checkpoints from [Google Cloud](https://drive.google.com/drive/folders/1028Ky-bJU6rSBXIMR6tf0wRAAvxIS6xP?usp=sharing). The datasets have been well pre-processed.

3. You can readily observe the superiority of GLAF through the checkpoints we provide:

   ```shell
   bash ./test.sh
   ```

4. You can also retrain GLAFF independently:

   ```shell
   bash ./run.sh
   ```

## Main Result

We conduct comprehensive experiments on nine real-world benchmark datasets across five domains. The result demonstrates that GLAFF significantly improves the robust prediction capability of mainstream forecasting models.

![](./img/result.png)

## Note

We revise data processing strategies to prevent future information leakage and conduct validation experiments on two representative datasets.  The length of the prediction window is set to 192. The experimental MSE results, presented in the table below, highlight the effectiveness of GLAFF. **GLAFF effectively leverages global information represented by timestamps to supplement the limited information provided by local observations.** As the history window narrows, the advantages of GLAFF become increasingly pronounced. Notably, our dataset is not exactly the same as the widely adopted version originating from [TSLib](https://github.com/thuml/Time-Series-Library), as we corrected several inaccurate timestamps.

|   Method    | Hist | Informer |   + Ours   | DLinear |   + Ours   |  TimesNet  |   + Ours   | iTransformer |   + Ours   |
| :---------: | :--: | :------: | :--------: | :-----: | :--------: | :--------: | :--------: | :----------: | :--------: |
|   Traffic   |  96  |  0.8401  | **0.7027** | 0.6374  | **0.5104** |   0.6604   | **0.6034** |    0.4514    | **0.4292** |
|   Traffic   |  48  |  0.8195  | **0.7191** | 0.7902  | **0.5663** |   0.6913   | **0.6172** |    0.5146    | **0.4936** |
|   Traffic   |  24  |  0.8164  | **0.7561** |  0.875  | **0.6028** |   0.7295   | **0.6286** |    0.6024    | **0.5807** |
| Electricity |  96  |  0.4567  | **0.3437** | 0.2038  | **0.1753** | **0.2003** |   0.2032   |    0.1688    | **0.1675** |
| Electricity |  48  |  0.3904  | **0.3390** | 0.2438  | **0.1991** | **0.1967** |   0.1986   |    0.1933    | **0.1874** |
| Electricity |  24  |  0.4014  | **0.3456** | 0.2849  | **0.2318** |   0.2132   | **0.2112** |    0.2423    | **0.2212** |

## Citation

If you find this repo or our work useful for your research, please consider citing the paper:

```tex
@inproceedings{
  author    = {Chengsen Wang and Qi Qi and Jingyu Wang and Haifeng Sun and Zirui Zhuang and Jinming Wu and Jianxin Liao},
  title     = {Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective},
  booktitle = {Thirty-eighth Conference on Neural Information Processing Systems},
  year      = {2024},
}
```

## Contact

If you have any question, please contact [cswang@bupt.edu.cn]().
