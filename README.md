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

MSE:  225.81894 Corr:  0.40225329643953395 Acc:  0.6392616706528479

## TIPS

1. train separate model
2. flip
3. resnet 152
4. shorter line
5. batch norm after hidden unit didnt help
6. black and white
7. color augmentation