Tiny images representation and nearest neighbor classifier: Accuracy = 0.191 (1-NN)

Bag of SIFT representation and nearest neighbor classifier: Accuracy = 0.503 (Step size=3 in vl_dsift)

Bag of SIFT representation and linear SVM classifier: Accuracy = 0.649 (vocab_size=200, Step=2 in vl_dsift)

Extra credit:
1. Variation of vocab_size
2. Taking 100 random training and testing images
3. Add spatial information
4. Soft assignment of visual words to histogram bins


NOTE: 
1. My code runs in about 11 minutes on my 4GB RAM(Windows), so it should run on an 8 GB RAM in under 10 minutes.
2. Commented line 35 of proj4.m run('vlfeat/toolbox/vl_setup'). Since I am not supposed to submit the vlfeat folder, if the code is run directly from my submission, this line would not work.