# Visual Illusions

This was a hobby project carried out intermittently with Ryan Sweke. The hope
was to train a neural network which you would feed any suitable image and it
would return the original image but modified in such a way that it would induce
a visual illusion, more specifically a kind of motion illusion. 

The basis for this hope was
[this](https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00345/full?utm_source=G-BLO&utm_medium=WEXT&utm_campaign=ECO_FPSYG_20180427_AI-optical-illusion)
paper by Watanabe et al. where they used the
[Prednet](https://coxlab.github.io/prednet/) architecture from the
Coxlab and showed that the prednet trained on natural images would predict an
optical flow in the rotating snake illusion. Our idea was to combine such
a trained prednet with a "style transfer"-type optimization process that would
seeks the input that maximizes some feature of the network, such as its
predicted output or some target activations of the higher layers in the
prednet. In the end we didn't make it very far. 

This repo isn't clean, please contact me if you'd like to use it, since the raw
material won't be of much help.
