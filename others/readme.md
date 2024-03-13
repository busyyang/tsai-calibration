pyTsai是一个使用Tsai方法结合blender的校准方法。其中src中的代码大部分应该是来自Tsai本人所写，非线性优化部分使用了minpack进行的。

可参考：
https://github.com/Csega/pyTsai

https://icube-forge.unistra.fr/flarue/Larue.AcqPipe.public/-/tree/master/externals/tsai

TsaiCpp是使用pyTsai进行一些优化，并使用C++语言重新写的代码，非线性优化采用了[Xtinc/matrix](https://github.com/Xtinc/matrix)的实现。ThreeStepOptimization方法就是pyTsai中的那种，FiveStepOptimization方法是参考专利：

北京航空航天大学. 一种基于透视成像模型标定的C型臂图像校正方法:CN200910087257.4[P]. 2009-11-18.
