# ee148-project

## Model evaluation

### The best model, trained on the full training set using early stopping.  
=====TEST RESULT=====  
=====MSE=====  
Pinch:  239.3994 Clench:  143.06828 Poke:  301.9881 Palm:  154.80666  
Average:  209.81561  
=====Corr=====  
Pinch:  (0.44519651545645866, 1.1412833390005551e-08) Clench:  (0.36350534627975173, 4.827041908393934e-06) Poke:  (0.6062976510122324, 2.0049352995549724e-16) Palm:  (0.39317615534800676, 6.4558886969491e-07)  
Average:  0.4520439170241124  
=====Acc=====  
Pinch:  0.6933333333333334 Clench:  0.5933333333333334 Poke:  0.6466666666666666 Palm:  0.6466666666666666  
Average:  0.6449999999999999  


### Results using CV
==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip/out <==  
=====TEST RESULT=====  
=====MSE=====  
Pinch:  231.55605 Clench:  155.27417 Poke:  253.38432 Palm:  169.35658  
Average:  202.39278  
=====Corr=====  
Pinch:  (0.48384248988427636, 3.555864485171391e-10) Clench:  (0.37000840154431336, 3.15857430671608e-06) Poke:  (0.6148939296282525, 5.731105669473861e-17) Palm:  (0.3563396745332501, 7.621400500099152e-06)  
Average:  0.45627112389752306  
=====Acc=====  
Pinch:  0.6733333333333333 Clench:  0.6133333333333333 Poke:  0.6733333333333333 Palm:  0.6266666666666667
Average:  0.6466666666666667  
=====CV RESULT=====  
MSE:  210.41158 Corr:  0.4796253201691215 Acc:  0.6617647058823529  

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip_YKs_random_grayscale_jitter/out <==  
=====TEST RESULT=====  
=====MSE=====  
Pinch:  241.0931 Clench:  142.5714 Poke:  246.43735 Palm:  158.54124  
Average:  197.16077  
=====Corr=====  
Pinch:  (0.48373430061013134, 3.592771066568557e-10) Clench:  (0.3723577633996663, 2.7036964058348438e-06) Poke:  (0.6134977362348577, 7.042051169761695e-17) Palm:  (0.40675751439842367, 2.4021210572278324e-07)  
Average:  0.4690868286607698  
=====Acc=====  
Pinch:  0.6933333333333334 Clench:  0.6266666666666667 Poke:  0.7 Palm:  0.6133333333333333  
Average:  0.6583333333333333  
=====CV RESULT=====  
MSE:  208.75655 Corr:  0.4833375016742556 Acc:  0.6641176470588235  

==> results/ckpt_with_augmented_data_but_overfitting/out <==  
=====TEST RESULT=====  
=====MSE=====  
Pinch:  247.24742 Clench:  165.04437 Poke:  282.40253 Palm:  170.39432  
Average:  216.27216  
=====Corr=====  
Pinch:  (0.43430687051620254, 2.8125640624846293e-08) Clench:  (0.3947423118585224, 5.773145665687315e-07) Poke:  (0.5808635594568557, 6.591346390977798e-15) Palm:  (0.33843364324324665, 2.2757764338718124e-05)  
Average:  0.43708659626870683  
=====Acc=====  
Pinch:  0.6466666666666666 Clench:  0.64 Poke:  0.6866666666666666 Palm:  0.6466666666666666  
Average:  0.6549999999999999  
=====CV RESULT=====  
MSE:  112.47374 Corr:  0.7698354025494265 Acc:  0.819294127160811  

==> results/ckpt_with_augmented_data_fixed_CV/out <==  
=====TEST RESULT=====  
=====MSE=====  
Pinch:  257.86746 Clench:  159.41824 Poke:  282.1695 Palm:  190.72478  
Average:  222.545  
=====Corr=====  
Pinch:  (0.42813423885296187, 4.624846599216912e-08) Clench:  (0.39674394465253704, 5.000424599897605e-07) Poke:  (0.5731640911377225, 1.7891768498029464e-14) Palm:  (0.2698979990902996, 0.0008374302893589074)  
Average:  0.41698506843338023  
=====Acc=====  
Pinch:  0.64 Clench:  0.64 Poke:  0.6733333333333333 Palm:  0.62  
Average:  0.6433333333333333  
=====CV RESULT=====  
MSE:  223.94478 Corr:  0.4431975601220161 Acc:  0.6467055142390712  


### Cross validation results before May 7
==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_pixels_1hidden_2pathways_separated_classifiers_direct_regression_with_horizontal_and_vertical_flip/out <==  
MSE:  220.29477 Corr:  0.45513356406767747 Acc:  0.6523529411764705

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip/out <==  
MSE:  213.05974 Corr:  0.46402621768133934 Acc:  0.6552941176470587

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_clench_only_with_horizontal_and_vertical_flip/out <==  
MSE:  164.03946 Corr:  0.430432086883019 Acc:  0.6494117647058824

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_maxPooling_separated_classifiers_with_horizontal_and_vertical_flip/out <==  
MSE:  244.02348 Corr:  0.3401501915583223 Acc:  0.6073529411764707

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_palm_only_with_horizontal_and_vertical_flip/out <==  
MSE:  184.09875 Corr:  0.3554128707693569 Acc:  0.6317647058823529

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_pinch_only_with_horizontal_and_vertical_flip/out <==  
MSE:  234.79407 Corr:  0.4847802803532083 Acc:  0.6670588235294118

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_poke_only_with_horizontal_and_vertical_flip/out <==  
MSE:  254.06868 Corr:  0.6115354463439585 Acc:  0.6988235294117646

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_16dim_output_with_horizontal_and_vertical_flip/out <==  
MSE:  217.04114 Corr:  0.4635804025188556 Acc:  0.6511764705882352

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_20dim_output_with_horizontal_and_vertical_flip/out <==  
MSE:  213.31563 Corr:  0.47235944238873967 Acc:  0.661470588235294

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_8dim_output_with_horizontal_and_vertical_flip/out <==  
MSE:  216.13884 Corr:  0.46234226237310827 Acc:  0.6479411764705882

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip_grayscale_jitter/out <==  
MSE:  216.95108 Corr:  0.4571144565366663 Acc:  0.6523529411764706

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip_grayscale/out <==  
MSE:  218.07687 Corr:  0.4633254609257218 Acc:  0.668235294117647

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip_jitter/out <==  
MSE:  222.70255 Corr:  0.43002600428523347 Acc:  0.6461764705882352

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip/out <==  
MSE:  206.81396 Corr:  0.48054613223572085 Acc:  0.6597058823529411

==> results/ckpt_resnet152_fulltune_with_normalized_log_long_normalized_log_short_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip_YKs_random_grayscale_jitter/out <==  
MSE:  215.72067 Corr:  0.46920354841849266 Acc:  0.6670588235294117


### Before April 24

==> results/ckpt_resnet152_fulltune_1hidden/out <==  
MSE:  228.28728 Corr:  0.41763258054849606 Acc:  0.6294117647058823

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_1hidden/out <==  
MSE:  227.03003 Corr:  0.4177470810816438 Acc:  0.6308823529411764

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_clench_only_model/out <==  
MSE:  175.71594 Corr:  0.39519955945912577 Acc:  0.648235294117647

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_fully_separated_model/out <==  
MSE:  217.6549175 Corr:  0.4567661156 Acc:  0.66

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways/out <==  
MSE:  225.25566 Corr:  0.44107836033636644 Acc:  0.6373529411764706

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_palm_only_model/out <==  
MSE:  169.87923 Corr:  0.38983390476760993 Acc:  0.6505882352941177

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_pinch_only_model/out <==  
MSE:  239.1064 Corr:  0.4807189304668487 Acc:  0.6670588235294118

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_poke_only_model/out <==  
MSE:  285.9181 Corr:  0.5613120678429663 Acc:  0.6741176470588236

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_2048hidden_units/out <==  
MSE:  222.46655 Corr:  0.42728767227017855 Acc:  0.6411764705882353

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_4096hidden_units_bn/out <==  
MSE:  248.90186 Corr:  0.3859441218439143 Acc:  0.6305882352941177

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers/out <==  
MSE:  220.02332 Corr:  0.44265670174882865 Acc:  0.6505882352941177

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_180rotation/out <==  
MSE:  222.46243 Corr:  0.44545438294941625 Acc:  0.6326470588235293

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_and_vertical_flip/out <==  
MSE:  212.47531 Corr:  0.47111907978685413 Acc:  0.655

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_horizontal_flip/out <==  
MSE:  212.19319 Corr:  0.4802025929831877 Acc:  0.6597058823529411

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_with_random_perspective/out <==  
MSE:  216.72836 Corr:  0.47391685451249 Acc:  0.6552941176470588

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden_2pathways_separated_classifiers_wo_augmentation/out <==  
MSE:  209.38864 Corr:  0.47873130037738176 Acc:  0.6526470588235294

==> results/ckpt_resnet152_fulltune_with_normalized_log_size_normalized_log_pixels_1hidden/out <==  
MSE:  224.4345 Corr:  0.43917454003652845 Acc:  0.6473529411764705

==> results/ckpt_resnet152_fulltune_with_size_1hidden_0.004lr/out <==  
MSE:  239.97849 Corr:  0.3339174221677915 Acc:  0.6052941176470588

==> results/ckpt_resnet152_fulltune_with_size_1hidden_1024hidden_units/out <==  
MSE:  231.36543 Corr:  0.39849500870123056 Acc:  0.6367647058823529

==> results/ckpt_resnet152_fulltune_with_size_1hidden_2048hidden_units/out <==  
MSE:  235.34937 Corr:  0.4043432316553092 Acc:  0.6270588235294118

==> results/ckpt_resnet152_fulltune_with_size_1hidden_8192hidden_units/out <==  
MSE:  232.96707 Corr:  0.4003600595522423 Acc:  0.6358823529411765

==> results/ckpt_resnet152_fulltune_with_size_1hidden/out <==  
MSE:  225.40027 Corr:  0.4163178369893578 Acc:  0.6344117647058823

==> results/ckpt_resnext101_32x8d_finetune_1hidden/out <==  
MSE:  230.48459 Corr:  0.36522075172100443 Acc:  0.6305882352941177

==> results/ckpt_resnext101_32x8d_finetune_2hidden/out <==  
MSE:  230.51079 Corr:  0.357980932460581 Acc:  0.6197058823529412

==> results/ckpt_resnext101_32x8d_fulltune_1hidden/out <==  
MSE:  230.8886 Corr:  0.4090679683754438 Acc:  0.6417647058823529

==> results/ckpt_resnext101_32x8d_fulltune_2hidden/out <==  
MSE:  233.15918 Corr:  0.3941256595745008 Acc:  0.6273529411764706

==> results/ckpt_resnext101_32x8d_fulltune_with_size_1hidden/out <==  
MSE:  229.62581 Corr:  0.4117277751502407 Acc:  0.635

==> results/ckpt_resnext101_32x8d_fulltune_with_size_2hidden/out <==  
MSE:  235.06213 Corr:  0.36037127285477416 Acc:  0.6173529411764707

==> results/ckpt_vgg16_finetune_2hidden/out <==  
MSE:  234.77205 Corr:  0.3622715843051389 Acc:  0.6205882352941177

==> results/ckpt_vgg16_finetune_with_size_2hidden/out <==  
MSE:  228.68689 Corr:  0.3758246133883908 Acc:  0.6276470588235294

==> results/ckpt_vgg16_fulltune_2hidden/out <==  
MSE:  234.37846 Corr:  0.3846657600522622 Acc:  0.6329411764705882

==> results/ckpt_vgg16_fulltune_with_size_2hidden/out <==  
MSE:  229.52426 Corr:  0.3842511294581498 Acc:  0.6255882352941177
