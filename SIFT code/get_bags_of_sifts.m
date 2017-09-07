% Starter code prepared by James Hays for Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_sifts(image_paths)
fprintf('Entry get_bags_of_sifts\n');
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html

%}

load('vocab.mat')
N=size(image_paths,1);
vocab_size = size(vocab, 2);
K=size(vocab,1);
vocab_T=vocab.';
%image_feats=0;

%UNCOMMENT FOR NORMAL OPERATION
image_feats=zeros(N,K);

%COMMENT THIS-It's for extra credit
%image_feats=zeros(N,K*5);
%histogram=zeros(1,vocab_size);
for i=1:N
    fprintf('i=%d\n',i);
    image=single(imread(char(image_paths(i))));
    
    %For extra credit-spatial information
%     [image_i,image_j]=size(image);
%      half_i=floor(image_i/2);
%      half_j=floor(image_j/2);
%      quart_i=floor(image_i/4);
%      quart_j=floor(image_j/4);
    %get dense SIFT features of each image
    [locations, SIFT_feat] = vl_dsift(image,'STEP',10); 
    %find distances to each cluster centre
    Distances = vl_alldist2(vocab_T,single(SIFT_feat));
    %assign to nearest cluster centre
    %c=cluster assignemnt list
    
    %%%UNCOMMENT FOR NORMAL OPERATION
    [minD,c]=min(Distances);  %c=1Xvocab_size
    %increment count for cluster in histogram
     binranges=1:K;
    [histogram]=histc(c,binranges); %%for full image
    
    %Extra credit- weighted histogram
%     norm_d=Distances./repmat(sum(Distances,1),K,1);
%     weights=(norm_d).^-2;  %1/d^2
%     norm_wts=weights./repmat(sum(weights,1),K,1);
%     histogram=sum(norm_wts,2);
    
    %%Extra credit- spatial information
    
%     quadrant1=find(locations(1,:)<=half_j & locations(2,:)<=half_i);
%     quadrant2=find(locations(1,:)>half_j & locations(2,:)<=half_i);
%     quadrant3=find(locations(1,:)<=half_j & locations(2,:)>half_i);
%     quadrant4=find(locations(1,:)>half_j & locations(2,:)>half_i);
%     [histogram1]=histc(c(quadrant1),binranges);
%     [histogram2]=histc(c(quadrant2),binranges);
%     [histogram3]=histc(c(quadrant3),binranges);
%     [histogram4]=histc(c(quadrant4),binranges);
%     histogram=cat(2,histogram,histogram1);
%     histogram=cat(2,histogram,histogram2);
%     histogram=cat(2,histogram,histogram3);
%     histogram=cat(2,histogram,histogram4);
%     
    %creating 4X4 grids
%     quadrant1=find(locations(1,:)<=quart_j & locations(2,:)<=quart_i);
%     quadrant2=find(locations(1,:)>quart_j & locations(1,:)<=half_j & locations(2,:)<=quart_i);
%     quadrant3=find(locations(1,:)>half_j & locations(1,:)<=3*quart_j & locations(2,:)<=quart_i);
%     quadrant4=find(locations(1,:)>3*quart_j & locations(1,:)<=image_j & locations(2,:)<=quart_i);
%     [histogram1]=histc(c(quadrant1),binranges);
%     [histogram2]=histc(c(quadrant2),binranges);
%     [histogram3]=histc(c(quadrant3),binranges);
%     [histogram4]=histc(c(quadrant4),binranges);
%     histogram=cat(2,histogram,histogram1);
%     histogram=cat(2,histogram,histogram2);
%     histogram=cat(2,histogram,histogram3);
%     histogram=cat(2,histogram,histogram4);
%     
%     quadrant1=find(locations(1,:)<=quart_j & locations(2,:)>quart_i & locations(2,:)<=half_i);
%     quadrant2=find(locations(1,:)>quart_j & locations(1,:)<=half_j & locations(2,:)>quart_i & locations(2,:)<=half_i);
%     quadrant3=find(locations(1,:)>half_j & locations(1,:)<=3*quart_j & locations(2,:)>quart_i & locations(2,:)<=half_i);
%     quadrant4=find(locations(1,:)>3*quart_j & locations(1,:)<=image_j & locations(2,:)>quart_i & locations(2,:)<=half_i);
%     [histogram1]=histc(c(quadrant1),binranges);
%     [histogram2]=histc(c(quadrant2),binranges);
%     [histogram3]=histc(c(quadrant3),binranges);
%     [histogram4]=histc(c(quadrant4),binranges);
%     histogram=cat(2,histogram,histogram1);
%     histogram=cat(2,histogram,histogram2);
%     histogram=cat(2,histogram,histogram3);
%     histogram=cat(2,histogram,histogram4);
%     
%     quadrant1=find(locations(1,:)<=quart_j & locations(2,:)>half_i & locations(2,:)<=3*quart_i);
%     quadrant2=find(locations(1,:)>quart_j & locations(1,:)<=half_j & locations(2,:)>half_i & locations(2,:)<=3*quart_i);
%     quadrant3=find(locations(1,:)>half_j & locations(1,:)<=3*quart_j & locations(2,:)>half_i & locations(2,:)<=3*quart_i);
%     quadrant4=find(locations(1,:)>3*quart_j & locations(1,:)<=image_j & locations(2,:)>half_i & locations(2,:)<=3*quart_i);
%     [histogram1]=histc(c(quadrant1),binranges);
%     [histogram2]=histc(c(quadrant2),binranges);
%     [histogram3]=histc(c(quadrant3),binranges);
%     [histogram4]=histc(c(quadrant4),binranges);
%     histogram=cat(2,histogram,histogram1);
%     histogram=cat(2,histogram,histogram2);
%     histogram=cat(2,histogram,histogram3);
%     histogram=cat(2,histogram,histogram4);
%     
%     quadrant1=find(locations(1,:)<=quart_j & locations(2,:)>3*quart_i & locations(2,:)<=image_i);
%     quadrant2=find(locations(1,:)>quart_j & locations(1,:)<=half_j & locations(2,:)>3*quart_i & locations(2,:)<=image_i);
%     quadrant3=find(locations(1,:)>half_j & locations(1,:)<=3*quart_j & locations(2,:)>3*quart_i & locations(2,:)<=image_i);
%     quadrant4=find(locations(1,:)>3*quart_j & locations(1,:)<=image_j & locations(2,:)>3*quart_i & locations(2,:)<=image_i);
%     [histogram1]=histc(c(quadrant1),binranges);
%     [histogram2]=histc(c(quadrant2),binranges);
%     [histogram3]=histc(c(quadrant3),binranges);
%     [histogram4]=histc(c(quadrant4),binranges);
%     histogram=cat(2,histogram,histogram1);
%     histogram=cat(2,histogram,histogram2);
%     histogram=cat(2,histogram,histogram3);
%     histogram=cat(2,histogram,histogram4);
    
    
    %image_orig=image;
%     for r=1:4
%         for s=1:4
%         if l==1; image=image_orig(1:half_i,1:half_j);
%         elseif l==2; image=image_orig(1:half_i,half_j+1:image_j);
%         elseif l==3; image=image_orig(half_i+1:image_i,1:half_j);
%         else image=image_orig(half_i+1:image_i,half_j+1:image_j);
%         end
%     quadrant=find(locations(1,:)>=(s-1)/4*image_j+1 & locations(1,:)<(s/4)*image_j & locations(2,:)>=(r-1)/4*image_i+1 & locations(2,:)>=(r/4)*image_i);
%     [histogram1]=histc(c(quadrant),binranges);
%      histogram=cat(2,histogram,histogram1);
    %     quadrant2=find(locations(1,:)>half_j & locations(2,:)<=half_i);
%     quadrant3=find(locations(1,:)<=half_j & locations(2,:)>half_i);
%     quadrant4=find(locations(1,:)>half_j & locations(2,:)>half_i);
%     [histogram1]=histc(c(quadrant),binranges);
%     [histogram2]=histc(c(quadrant2),binranges);
%     [histogram3]=histc(c(quadrant3),binranges);
%     [histogram4]=histc(c(quadrant4),binranges);
%     histogram=cat(2,histogram,histogram1);
%     histogram=cat(2,histogram,histogram2);
%     histogram=cat(2,histogram,histogram3);
%     histogram=cat(2,histogram,histogram4);
%         end
%     end
    %histogram=cat(2,histogram,zeros(4*vocab_size,1));
    
       
%     
%     [histogram,y]=hist(c,unique(c));
%     %It's possible we have 0 value for the last cluster centre, in which
%     %case this will be a 1Xvocab_size-1 array
%     %making sure it's the right size
%     if vocab_size ~= size(histogram,2)
%         m=vocab_size-size(histogram,2);
%         histogram=cat(2,histogram,zeros(1,m));
%     end

    %normalise histogram
    histogram=histogram/sum(histogram);
    image_feats(i,:)=histogram(:);
%     if image_feats == 0
%         image_feats=histogram;
%     else
%         %image_feats is an N x vocab_size matrix
%         image_feats=cat(1,image_feats,histogram);
%     end   
end




