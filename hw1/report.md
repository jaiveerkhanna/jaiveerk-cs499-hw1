Key Implementation Choices:

Structure: Embedding Layer --> LSTM Layer --> (2 x FC Layer [one for each classification head])

I chose this structure because I thought it made sense to keep one model. When we discussed in class, the advice was only to split model if we thought the classification heads were fully independent, however I felt like there was definitely going to be some relation between an action and a target. For example, objects might be more likely to be operated on, locations more likely to be travelled to. At first I was a bit confused on how to tackle this, but I feel like as I implemented it made a lot more sense. We need the seperate final FC's because they need to map to different classifications, but its beneficial to use the same LSTM layer so that we can try to uncover relationships between the two classification heads.

Key Variables:
    Learning Rate: 0.005 --> previous lr given was very small and model was not converging, I purposefully did not use 0.01 because it felt like too large a step

    Number of epochs: 1000 --> stuck to default, this is closely related to learning rate and i felt i wanted at least 1000 reps. Because performance was good I felt no need to change. Also convergence tended to happen towards the end for multiple different trials (visualization.png shows this well and is located in the repo)

    Episodes of data considered: first 50 --> this is simply because it took too long to run the entire dataset through. If you would like to try, go to the line 48 of train.py and remove the count variable / if statement that stops input processing after 50 counts. I tried running the code on other subsets of episodes (keeping size 50) and achieved similar results so am happy with this. 

    Embedding / Hidden Dimension layer: I chose these to be the same size of 20. When I was training, I used dim of 2 just to keep things fast but i felt that ultimately when going for performance this was too small a dimension. I'm happy with performance so will leave it at this

    Minibatch size: 32 --> stuck to default because this was working well

    Optimizer: For my optimizer, I stuck to what we have been using in class which was stochasting gradient descent. This worked well and in full honesty since I am new to AI I have limited exposure to other optimizers. I explored some of the ones in the torch library but ultimately thought that with the performance I was getting it was better to stick to an optimizer I understood thoroughly so I could troubleshoot / pick other hyperparameters effectively

    Loss functions: for both classification heads, I used cross entropy loss. I summed both cross entropy loss scores to calculate total loss. Cross Entropy loss was what we have covered well in class and helped the data converge better. This was standard in a lot of the models I saw online and served me well. 

Performance:

Based on the hyperparameters given we achieved the following results (taken from terminal):
    train action loss : 3.5502268224954605 | train target loss: 2.9103443324565887
    train action acc : 0.8730650154798761  | train target acc: 0.9009287925696594

A snapshot of what the val performance was like (at the 995th epoch):
    val action loss : 12.274666622281075 | val target loss: 11.113841012120247
    val action acc : 0.8802163833075735 | val target losaccs: 0.91112828438949

A visualization of this performance over time is included as visualization.png in this directory

