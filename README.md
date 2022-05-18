# MCG-VITON — Official Implementation
![](https://github.com/NerdFNY/MCG-VITON/raw/master/fig.png)  
This is the official implementation of paper "Toward Multi-Category Garments Virtual Try-on Method by Coarse to Fine TPS Deformation" <br>

## 
- Our paper has been accepted by NCAA, and it is [available online](https://link.springer.com/article/10.1007/s00521-022-07173-w).

## Requirements
- python3
- numpy
- cv2
- skimage
- matplotlib

## Getting Started

We have provided an example for preparing your dateset.  
You can change coefficients in options.py and run.

``` 
python main.py
```

## Example
![](https://github.com/NerdFNY/MCG-VITON/raw/master/example.png)  

## Citation
If you find our work useful in your research, please consider citing our paper:

```
@article{fang2022toward,  
  title={Toward multi-category garments virtual try-on method by coarse to fine TPS deformation},  
  author={Fang, Naiyu and Qiu, Lemiao and Zhang, Shuyou and Wang, Zili and Hu, Kerui and Li, Heng},  
  journal={Neural Computing and Applications},  
  pages={1--19},  
  year={2022},  
  publisher={Springer}  
}
```
## Acknowledgments
we have leveraged [SMPL](https://smpl.is.tue.mpg.de/) as our virtual 3d human model  
Thanks to [the TPS implementation](https://github.com/iwyoo/TPS_STN-tensorflow)
