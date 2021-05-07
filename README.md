# TEST RESULTS

## VGG16

* FC: 4096 -> 4096 -> 10

5-fold average:

MSE:  232.39531 Corr:  0.39045113092274797 Acc:  0.638990602158023


## VGG11

* FC: 4096 -> 129 -> 10

4-fold average:

MSE:  237.5575 Corr:  0.37037888073722564 Acc:  0.6216062095845514

## VGG11_Size

* FC: 4096 -> 128 + size -> 10

4-fold average:

MSE:  233.90048 Corr:  0.3804844526806619 Acc:  0.621321086898751

## VGG11_SIZE with color transform and without rotation/perspective

* FC: 4096 -> 128 + size -> 10
* color transform

4-fold average:

MSE:  222.43243 Corr:  0.4064351394165007 Acc:  0.6351578417043139

MSE:  225.31706 Corr:  0.3864094993978276 Acc:  0.6251522499778546

### With random rotation (30)

MSE:  231.561 Corr:  0.37659053483302374 Acc:  0.6118677473646914

## VGG11 large

* FC: 4096 -> 2048 -> 10
* color transform

4-fold average:

MSE:  229.46928 Corr:  0.3687010674491655 Acc:  0.6157238240765347

## VGG11 no grad

* no grad in VGG11
* FC: 4096 -> 128 + size -> 10
* color transform

4-fold average:

MSE:  227.12564 Corr:  0.3889142380582102 Acc:  0.6321875830454424

## VGG11 size color familiarity

* FC: 4096 -> 128 + size + familiarty -> 10
* color transform

4-fold average:

MSE:  229.72495 Corr:  0.38814254789824104 Acc:  0.6289432744264328

## VGG11 size familiarity wh_ratio

* FC: 4096 -> 128 + size + familiarity + wh_ratio -> 10
* color transform

MSE: 227.66507 Corr: 0.3916259364769663 Acc: 0.62247541854903

## VGG11 size wh_ratio

* FC: 4096 -> 128 + size + wh_ratio -> 10
* color transform
* MSE:  227.96414 Corr:  0.39071493091865883 Acc:  0.6271854792275666


## RESNET 152

MSE:  214.79027 Corr:  0.45263997564423714 Acc:  0.656381488617238

## TIPS

1. train separate model
2. flip
3. resnet 152
4. shorter line
5. batch norm after hidden unit didnt help
6. black and white
7. color augmentation