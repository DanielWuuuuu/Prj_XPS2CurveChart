## Convert XPS/PDF to Curve Chart Image

Our files are categorized into four frequencies, namely *_4kHz.xps, *_2kHz.xps, *_1kHz.xps, and *_500Hz.xps. While these files typically contain a curve chart, it lack the corresponding data. Our objective is to obtain the data, calculate each frequency's mean data, and generate a graph.

This project comprises three ipynb files and one py file, and we kindly request that you follow the instructions below to execute them.

#### Prj_XPS2CurveChart_01.ipynb

- This part is convert xps/pdf file to the data of curve chart
  
  xps file looks like this:
  
  ![](docs\xps_exemple.png)

- Modify global variable: `ROOT` 
  
  The files you want to convert are all in the ROOT

- Before excute this ipynb, the dir tree like:
  
  ```
  ROOT  
  ├─ 00001_R.xps  
  ├─ 00002_L_500-2000.xps  
  ├─ 00002_L_4000.xps  
  └─ ..  
  ```

- After excute this ipynb, the dir tree like:
  
  ```
  ROOT  
  ├─ 00001_R  
  │  ├─ 1kHz  
  │  │  ├─ 00001_R_1kHz.png  
  │  │  ├─ out_00001_R_1kHz.csv  
  │  │  ├─ out_00001_R_1kHz.png  
  │  │  └─ plot_00001_R_1kHz.png  
  │  ├─ 2kHz  
  │  │  ├─ 00001_R_1kHz.png  
  │  │  ├─ out_00001_R_1kHz.csv  
  │  │  ├─ out_00001_R_1kHz.png  
  │  │  └─ plot_00001_R_1kHz.png  
  │  ├─ 4kHz  
  │  │  └─ ..  
  │  ├─ 500Hz  
  │  │  └─ ..  
  │  └─ 00001_R.xps  
  ├─ 00002_L_500-2000  
  │  ├─ 1kHz  
  │  │  └─ ..  
  │  ├─ 2kHz  
  │  │  └─ ..  
  │  ├─ 500Hz  
  │  │  └─ ..  
  │  └─ 00002_L_500-2000.xps  
  ├─ 00002_L_4000  
  │  ├─ 4kHz  
  │  │  └─ ..  
  │  └─ 00002_L_4000.xps  
  └─ ..  
  ```

#### Prj_XPS2CurveChart_02.ipynb

- This part is calculate each frequency's mean data, and generate a graph in the `output` folder

- Modify global variable: `ROOT` 
  
  The files you want to calculate are all in the ROOT

- Before excute this ipynb, the dir tree like:
  
  ```
  ROOT  
  ├─ 00001_R  
  │  ├─ 1kHz  
  │  │  └─ ..  
  │  ├─ 2kHz  
  │  │  └─ ..  
  │  ├─ 4kHz  
  │  │  └─ ..  
  │  ├─ 500Hz  
  │  │  └─ ..  
  │  └─ 00001_R.xps  
  ├─ 00002_L_500-2000  
  │  ├─ 1kHz  
  │  │  └─ ..  
  │  ├─ 2kHz  
  │  │  └─ ..  
  │  ├─ 500Hz  
  │  │  └─ ..  
  │  └─ 00002_L_500-2000.xps  
  ├─ 00002_L_4000  
  │  ├─ 4kHz  
  │  │  └─ ..  
  │  └─ 00002_L_4000.xps  
  └─ ..  
  ```

- After excute this ipynb, the dir tree like:
  
  ```
  ROOT  
  ├─ 00001_R  
  │  ├─ 1kHz  
  │  │  └─ ..  
  │  ├─ 2kHz  
  │  │  └─ ..  
  │  ├─ 4kHz  
  │  │  └─ ..  
  │  ├─ 500Hz  
  │  │  └─ ..  
  │  └─ 00001_R.xps  
  ├─ 00002_L_500-2000  
  │  ├─ 1kHz  
  │  │  └─ ..  
  │  ├─ 2kHz  
  │  │  └─ ..  
  │  ├─ 500Hz  
  │  │  └─ ..  
  │  └─ 00002_L_4000.xps  
  ├─ 00002_L_4000  
  │  ├─ 4kHz  
  │  │  └─ ..  
  │  └─ 00002_L_4000.xps  
  ├─ .. 
  └─ output  
     ├─ out_1kHz.csv  
     ├─ out_1kHz.png  
     ├─ out_2kHz.csv  
     ├─ out_2kHz.png  
     ├─ out_4kHz.csv  
     ├─ out_4kHz.png  
     ├─ out_500Hz.csv  
     └─ out_500Hz.png  
  ```

#### Prj_XPS2CurveChart_03.ipynb

- This part is plot data of the same frequency together

- Modify global variable: `ROOT`
  
  The files you want to plot together are all in the ROOT

- Before excute this ipynb, the dir tree like:
  
  ```
  ROOT  
  ├─ NH  
  │  ├─ ..  
  │  └─ output  
  ├─ NHT  
  │  ├─ ..  
  │  └─ output  
  └─ Normal  
     ├─ ..  
     └─ output  
  ```

- After excute this ipynb, the dir tree like:
  
  ```
  ROOT  
  ├─ NH  
  │  ├─ ..  
  │  └─ output  
  ├─ NHT  
  │  ├─ ..  
  │  └─ output  
  ├─ Normal  
  │  ├─ ..  
  │  └─ output  
  └─ output  
     ├─ out_1kHz.png  
     ├─ out_2kHz.png  
     ├─ out_4kHz.png  
     └─ out_500Hz.png  
  ```
