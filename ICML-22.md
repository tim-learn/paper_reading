> Oral: Online Learning for Min Sum Set Cover and Pandora’s Box
> 
> Authors: Evangelia Gergatsouli and Christos Tzamos
> 
> Abstract: Two central problems in Stochastic Optimization are Min-Sum Set Cover and Pandora’s Box. In Pandora’s Box, we are presented with n boxes, each containing an unknown value and the goal is to open the boxes in some order to minimize the sum of the search cost and the smallest value found. Given a distribution of value vectors, we are asked to identify a near-optimal search order. Min-Sum Set Cover corresponds to the case where values are either 0 or infinity.In this work, we study the case where the value vectors are not drawn from a distribution but are presented to a learner in an online fashion. We present a computationally efficient algorithm that is constant-competitive against the cost of the optimal search order. We extend our results to a bandit setting where only the values of the boxes opened are revealed to the learner after every round. We also generalize our results to other commonly studied variants of Pandora’s Box and Min-Sum Set Cover that involve selecting more than a single value subject to a matroid constraint.


> Oral: Adapting to Mixing Time in Stochastic Optimization with Markovian Data
> 
> Authors: Ron Dorfman and Kfir Levy
> 
> Abstract: We consider stochastic optimization problems where data is drawn from a Markov chain. Existing methods for this setting crucially rely on knowing the mixing time of the chain, which in real-world applications is usually unknown. We propose the first optimization method that does not require the knowledge of the mixing time, yet obtains the optimal asymptotic convergence rate when applied to convex problems. We further show that our approach can be extended to: (i) finding stationary points in non-convex optimization with Markovian data, and (ii) obtaining better dependence on the mixing time in temporal difference (TD) learning; in both cases, our method is completely oblivious to the mixing time. Our method relies on a novel combination of multi-level Monte Carlo (MLMC) gradient estimation together with an adaptive learning method.


> Oral: LIDL: Local Intrinsic Dimension estimation using approximate Likelihood
> 
> Authors: Piotr Tempczyk and Rafał Michaluk and Łukasz Garncarek and Adam Golinski and Przemysław Spurek and Jacek Tabor
> 
> Abstract: Understanding how neural networks work is one of the most important questions in machine learning research. Their performance is connected with the shape of the data manifold, and the structure of this manifold can be explored with local intrinsic dimension (LID) estimation methods. Unfortunately, they do not scale well to high-dimensional datasets used with neural networks and give inaccurate estimates for complex manifolds. We address those challenges by proposing a new method - LIDL - that uses novel normalizing flow models. Our method yields accurate estimates on complex manifolds and scales well to problems with thousands of dimensions. We use our algorithm to briefly show, that LID is connected with neural networks performance in supervised and unsupervised settings.


> Oral: Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them
> 
> Authors: Florian Tramer
> 
> Abstract: Making classifiers robust to adversarial examples is challenging. Thus, many works tackle the seemingly easier task of \emph{detecting} perturbed inputs.We show a barrier towards this goal. We prove a \emph{hardness reduction} between detection and classification of adversarial examples: given a robust detector for attacks at distance $\epsilon$ (in some metric), we show how to build a similarly robust (but inefficient) \emph{classifier} for attacks at distance $\epsilon/2$.Our reduction is \emph{computationally} inefficient, but preserves the \emph{data complexity} of the original detector. The reduction thus cannot be directly used to build practical classifiers.Instead, it is a useful sanity check to test whether empirical detection results imply something much stronger than the authors presumably anticipated (namely a highly robust and data-efficient \emph{classifier}).To illustrate, we revisit $14$ empirical detector defenses published over the past years. For $12/14$ defenses, we show that the claimed detection results imply an inefficient classifier with robustness far beyond the state-of-the-art--- thus casting some doubts on the results' validity.Finally, we show that our reduction applies in both directions: a robust classifier for attacks at distance $\epsilon/2$ implies an inefficient robust detector at distance $\epsilon$. Thus, we argue that robust classification and robust detection should be regarded as (near)-equivalent problems, if we disregard their \emph{computational} complexity.

> Oral: Near-Exact Recovery for Tomographic Inverse Problems via Deep Learning
> 
> Authors: Martin Genzel and Ingo Gühring and Jan Macdonald and Maximilian März
> 
> Abstract: This work is concerned with the following fundamental question in scientific machine learning: Can deep-learning-based methods solve noise-free inverse problems to near-perfect accuracy? Positive evidence is provided for the first time, focusing on a prototypical computed tomography (CT) setup. We demonstrate that an iterative end-to-end network scheme enables reconstructions close to numerical precision, comparable to classical compressed sensing strategies. Our results build on our winning submission to the recent AAPM DL-Sparse-View CT Challenge. Its goal was to identify the state-of-the-art in solving the sparse-view CT inverse problem with data-driven techniques. A specific difficulty of the challenge setup was that the precise forward model remained unknown to the participants. Therefore, a key feature of our approach was to initially estimate the unknown fanbeam geometry in a data-driven calibration step. Apart from an in-depth analysis of our methodology, we also demonstrate its state-of-the-art performance on the open-access real-world dataset LoDoPaB CT.


> Oral: Re-evaluating Word Mover's Distance
> 
> Authors: Ryoma Sato and Makoto Yamada and Hisashi Kashima
> 
> Abstract: The word mover's distance (WMD) is a fundamental technique for measuring the similarity of two documents. As the crux of WMD, it can take advantage of the underlying geometry of the word space by employing an optimal transport formulation. The original study on WMD reported that WMD outperforms classical baselines such as bag-of-words (BOW) and TF-IDF by significant margins in various datasets. In this paper, we point out that the evaluation in the original study could be misleading. We re-evaluate the performances of WMD and the classical baselines and find that the classical baselines are competitive with WMD if we employ an appropriate preprocessing, i.e., L1 normalization. In addition, We introduce an analogy between WMD and L1-normalized BOW and find that not only the performance of WMD but also the distance values resemble those of BOW in high dimensional spaces.


> Oral: A Dynamical System Perspective for Lipschitz Neural Networks
> 
> Authors: Laurent Meunier and Blaise Delattre and Alexandre ARAUJO and Alexandre Allauzen
> 
> Abstract: The Lipschitz constant of neural networks has been established as a key quantity to enforce the robustness  to adversarial examples. In this paper, we tackle the problem of building $1$-Lipschitz Neural Networks. By studying Residual Networks from a continuous time dynamical system perspective, we provide a generic method to build $1$-Lipschitz Neural Networks and show that some previous approaches are special cases of this framework. Then, we extend this reasoning and show that ResNet flows derived from convex potentials define $1$-Lipschitz transformations, that lead us to define the {\em Convex Potential Layer} (CPL). A comprehensive set of experiments on several datasets demonstrates the scalability of our architecture and the benefits as an $\ell_2$-provable defense against adversarial examples.

> Oral: Scaling up Universal Methods for Convex Optimization
> 
> Authors: Kimon Antonakopoulos and Dong Quan Vu and Volkan Cevher and Kfir Levy and Panayotis Mertikopoulos
> 
> Abstract: Universal methods achieve optimal convergence rate guarantees in convex optimization without any prior knowledge of the problem's regularity parameters or the attributes of the gradient oracle employed by the method. In this regard, existing state-of-the-art algorithms achieve an $O(1/T^2)$ convergence rate in Lipschitz smooth problems with a perfect gradient oracle, and an $O(1/sqrt{T})$ convergence speed when the underlying problem is non-smooth and/or the gradient oracle is stochastic. On the downside, these methods do not take into account the dependence of these guarantees on the problem's dimensionality, and this can have a catastrophic impact on a method's convergence, in both theory and practice. Our paper aims to bridge this gap by providing a scalable universal method - dubbed UnDERGrad - which enjoys an almost dimension-free oracle complexity in problems with a favorable geometry (like the simplex, $\ell_1$-ball or trace-constraints), while retaining the order-optimal dependence on T described above. These "best of both worlds" guarantees are achieved via a primal-dual update scheme inspired by the dual exploration method for variational inequalities. 

> Oral: An Improved Analysis of Algorithmic Robustness
> 
> Authors: Kenji Kawaguchi and Kyle Luh and Jiaoyang Huang and Zhun Deng
> 
> Abstract: Algorithmic robustness is a powerful mathematical framework that relates the robustness of a learning algorithm to its expected loss and has been utilized for various learning algorithms. This study improves the framework of algorithmic robustness in two directions to address an open problem. The first is to reduce the dependence on the covering number. The second is the replacement of a maximum over the entire hypothesis space with the single hypothesis returned by the algorithm under a particular training dataset. We include several examples in which our bounds are provably preferable. Moreover, the experiments on real-world data and theoretical models demonstrate a near-exponential improvement in various situations. To achieve these improvements, we do not require additional assumptions on the unknown distribution; instead, we only incorporate an observable and computable property of the training samples. These improvements to the foundations of algorithmic robustness have consequences for numerous applications. A key technical innovation is an improved concentration bound for multinomial random variables that is of independent interest beyond algorithmic robustness.


> Oral: Random Gegenbauer Features for Scalable Kernel Methods
> 
> Authors: Insu Han and Amir Zandieh and Haim Avron
> 
> Abstract: We propose efficient random features for approximating a new and rich class of kernel functions that we refer to as Generalized Zonal Kernels (GZK). Our proposed GZK family, generalizes the zonal kernels (i.e., dot-product kernels on the unit sphere) by introducing radial factors in their Gegenbauer series expansion, and includes a wide range of ubiquitous kernel functions such as the entirety of dot-product kernels as well as the Gaussian and the recently introduced Neural Tangent kernels. Interestingly, by exploiting the reproducing property of the Gegenbauer polynomials, we can construct efficient random features for the GZK family based on randomly oriented Gegenbauer kernels. We prove subspace embedding guarantees for our Gegenbauer features which ensures that our features can be used for approximately solving learning problems such as kernel k-means clustering, kernel ridge regression, etc. Empirical results show that our proposed features outperform recent kernel approximation methods.


> Oral: Function-space Inference with Sparse Implicit Processes
> 
> Authors: Simon R Santana and Bryan Zaldivar and Daniel Hernandez-Lobato
> 
> Abstract: Implicit Processes (IPs) represent a flexible framework that can be used to describe a wide variety of models, from Bayesian neural networks, neural samplers and data generators to many others. IPs also allow for approximate inference in function-space. This change of formulation solves intrinsic degenerate problems of parameter-space approximate inference concerning the high number of parameters and their strong dependencies in large models. For this, previous works in the literature have attempted to employ IPs both to set up the prior and to approximate the resulting posterior. However, this has proven to be a challenging task. Existing methods that can tune the prior IP result in a Gaussian predictive distribution, which fails to capture important data patterns. By contrast, methods producing flexible predictive distributions by using another IP to approximate the posterior process cannot tune the prior IP to the observed data. We propose here the first method that can accomplish both goals. For this, we rely on an inducing-point representation of the prior IP, as often done in the context of sparse Gaussian processes. The result is a scalable method for approximate inference with IPs that can tune the prior IP parameters to the data, and that provides accurate non-Gaussian predictive distributions.


> Oral: Individual Preference Stability for Clustering
> 
> Authors: Saba Ahmadi and Pranjal Awasthi and Samir Khuller and Matthäus Kleindessner and Jamie Morgenstern and Pattara Sukprasert and Ali Vakilian
> 
> Abstract: In this paper, we propose a natural notion of individual preference (IP) stability for clustering, which asks that every data point, on average, is closer to the points in its own cluster than to the points in any other cluster. Our notion can be motivated from several perspectives, including game theory and algorithmic fairness. We study several questions related to our proposed notion. We first show that deciding whether a given data set allows for an IP-stable clustering in general is NP-hard. As a result we explore the design of efficient algorithms for finding IP-stable clusterings in some restricted metric spaces. We present a polytime algorithm to find a clustering satisfying exact IP-stability in one dimensions, and an efficient algorithm to find IP-stable 2-clustering for tree metrics. We also consider relaxing the stability constraint, i.e., every data point should not be too far from its own cluster, compared to any other clusters. In such a case, we show polytime algorithms with different guarantees. We evaluate our algorithms and several standard clustering approaches on real data sets. 


> Oral: A new similarity measure for covariate shift with applications to nonparametric regression
> 
> Authors: Reese Pathak and Cong Ma and Martin Wainwright
> 
> Abstract: We study covariate shift in the context of nonparametric regression. We introduce a new measure of distribution mismatch between the source and target distributions using the integrated ratio of probabilities of balls at a given radius. We use the scaling of this measure with respect to the radius to characterize the minimax rate of estimation over a family of Hölder continuous functions under covariate shift. In comparison to the recently proposed notion of transfer exponent, this measure leads to a sharper rate of convergence and is more fine-grained. We accompany our theory with concrete instances of covariate shift that illustrate this sharp difference. 


> Oral: Minimum Cost Intervention Design for Causal Effect Identification
> 
> Authors: Sina Akbari and Jalal Etesami and Negar Kiyavash
> 
> Abstract: Pearl’s do calculus is a complete axiomatic approach to learn the identifiable causal effects from observational data. When such an effect is not identifiable, it is necessary to perform a collection of often costly interventions in the system to learn the causal effect. In this work, we consider the problem of designing the collection of interventions with the minimum cost to identify the desired effect. First, we prove that this prob-em is NP-complete, and subsequently propose an algorithm that can either find the optimal solution or a logarithmic-factor approximation of it. This is done by establishing a connection between our problem and the minimum hitting set problem. Additionally, we propose several polynomial  time  heuristic  algorithms  to  tackle the computational complexity of the problem. Although these algorithms could potentially stumble on sub-optimal solutions, our simulations show that they achieve small regrets on random graphs.


> Oral: An Analytical Update Rule for General Policy Optimization
> 
> Authors: Hepeng Li and Nicholas Clavette and Haibo He
> 
> Abstract: We present an analytical policy update rule that is independent of parameterized function approximators. The update rule is suitable for optimizing general stochastic policies with monotonic improvement guarantee. The update rule is derived from a closed-form solution to trust-region policy optimization using calculus of variation, following a new theoretical result that tightens existing bounds for trust-region methods. The update rule builds a connection between policy-search methods and value-function methods. Also, an off-policy reinforcement learning algorithm can be derived naturally based on the policy update rule and the property of monotonic improvement guarantee remains. Furthermore, we prove that the update rule extends immediately to multi-agent systems when policy updates are performed by one agent at a time.


> Oral: Deletion Robust Submodular Maximization over Matroids
> 
> Authors: PAUL DUETTING and Federico Fusco and Silvio Lattanzi and Ashkan Norouzi-Fard and Morteza Zadimoghaddam
> 
> Abstract: Maximizing a monotone submodular function is a fundamental task in machine learning. In this paper we study the deletion robust version of the problem under the classic matroids constraint. Here the goal is to extract a small size summary of the dataset that contains a high value independent set even after an adversary deleted some elements. We present constant-factor approximation algorithms, whose space complexity depends on the rank $k$ of the matroid and the number $d$ of deleted elements. In the centralized setting we present a $(3.582+O(\varepsilon))$-approximation algorithm with summary size $O(k + \frac{d \log k}{\varepsilon^2})$. In the streaming setting we provide a $(5.582+O(\varepsilon))$-approximation algorithm with summary size and memory $O(k + \frac{d \log k}{\varepsilon^2})$. We complement our theoretical results with an in-depth experimental analysis showing the effectiveness of our algorithms on real-world datasets.

> Oral: Adversarially Trained Actor Critic for Offline Reinforcement Learning
> 
> Authors: Ching-An Cheng and Tengyang Xie and Nan Jiang and Alekh Agarwal
> 
> Abstract: We propose Adversarially Trained Actor Critic (ATAC), a new model-free algorithm for offline reinforcement learning (RL) under insufficient data coverage, based on a two-player Stackelberg game framing of offline RL: A policy actor competes against an adversarially trained value critic, who finds data-consistent scenarios where the actor is inferior to the data-collection behavior policy. We prove that, when the actor attains no regret in the two-player game, running ATAC produces a policy that provably 1) outperforms the behavior policy over a wide range of hyperparameters, and 2) competes with the best policy covered by data with appropriately chosen hyperparameters. Compared with existing works, notably our framework offers both theoretical guarantees for general function approximation and a deep RL implementation scalable to complex environments and large datasets. In the D4RL benchmark, ATAC consistently outperforms state-of-the-art offline RL algorithms on a range of continuous control tasks.


> Oral: Learning Mixtures of Linear Dynamical Systems
> 
> Authors: Yanxi Chen and H. Vincent Poor
> 
> Abstract: We study the problem of learning a mixture of multiple linear dynamical systems (LDSs) from unlabeled short sample trajectories, each generated by one of the LDS models. Despite the wide applicability of mixture models for time-series data, learning algorithms that come with end-to-end performance guarantees are largely absent from existing literature. There are multiple sources of technical challenges, including but not limited to (1) the presence of latent variables (i.e. the unknown labels of trajectories); (2) the possibility that the sample trajectories might have lengths much smaller than the dimension $d$ of the LDS models; and (3) the complicated temporal dependence inherent to time-series data. To tackle these challenges, we develop a two-stage meta-algorithm, which is guaranteed to efficiently recover each ground-truth LDS model up to error $\tilde{O}(\sqrt{d/T})$, where $T$ is the total sample size. We validate our theoretical studies with numerical experiments, confirming the efficacy of the proposed algorithm. 

> Oral: data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language
> 
> Authors: Alexei Baevski and Wei-Ning Hsu and Qiantong Xu and Arun Babu and Jiatao Gu and Michael Auli
> 
> Abstract: While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.


> Oral: Online Decision Transformer
> 
> Authors: Qinqing Zheng and Amy Zhang and Aditya Grover
> 
> Abstract: Recent work has shown that offline reinforcement learning (RL) can be formulated as a sequence modeling problem~\cite{chen2021decision,janner2021offline} and solved via approaches similar to large-scale language modeling. However, any practical instantiation of RL also involves an online component, where policies pretrained on passive offline datasets are finetuned via task-specific interactions with the environment. We propose Online Decision Transformers (ODT), an RL algorithm based on sequence modeling that blends offline pretraining with online finetuning in a unified framework. Our framework uses sequence-level entropy regularizers in conjunction with autoregressive modeling objectives for sample-efficient exploration and finetuning. Empirically, we show that ODT is competitive with the state-of-the-art in absolute performance on the D4RL benchmark but shows much more significant gains during the finetuning procedure.


> Oral: How Tempering Fixes Data Augmentation in Bayesian Neural Networks
> 
> Authors: Lorenzo Noci and Gregor Bachmann and Thomas Hofmann
> 
> Abstract: While Bayesian neural networks (BNNs) provide a sound and principled alternative to standard neural networks, an artificial sharpening of the posterior usually needs to be applied to reach comparable performance. This is in stark contrast to theory, dictating that given an adequate prior and a well-specified model, the untempered Bayesian posterior should achieve optimal performance. Despite the community's extensive efforts, the observed gains in performance still remain disputed with several plausible causes pointing at its origin. While data augmentation has been empirically recognized as one of the main drivers of this effect, a theoretical account of its role, on the other hand, is largely missing. In this work we identify two interlaced factors concurrently influencing the strength of the cold posterior effect, namely the correlated nature of augmentations and the degree of invariance of the employed model to such transformations. By theoretically analyzing simplified settings, we prove that tempering implicitly reduces the misspecification arising from modeling augmentations as i.i.d. data. The temperature mimics the role of the effective sample size, reflecting the gain in information provided by the augmentations. We corroborate our theoretical findings with extensive empirical evaluations, scaling to realistic BNNs. By relying on the framework of group convolutions, we experiment with models of varying inherent degree of invariance, confirming its hypothesized relationship with the optimal temperature.


> Oral: Generative Trees: Adversarial and Copycat
> 
> Authors: Richard Nock and Mathieu Guillame-Bert
> 
> Abstract: While Generative Adversarial Networks (GANs) achieve spectacular results on unstructured data like images, there is still a gap on \textit{tabular data}, data for which state of the art \textit{supervised learning} still favours decision tree (DT)-based models. This paper proposes a new path forward for the generation of tabular data, exploiting decades-old understanding of the supervised task's best components for DT induction, from losses (properness), models (tree-based) to algorithms (boosting). The \textit{properness} condition on the supervised loss -- which postulates the optimality of Bayes rule -- leads us to a variational GAN-style loss formulation which is \textit{tight} when discriminators meet a calibration property trivially satisfied by DTs, and, under common assumptions about the supervised loss, yields "one loss to train against them all" for the generator: the $\chi^2$. We then introduce tree-based generative models, \textit{generative trees} (GTs), meant to mirror on the generative side the good properties of DTs for classifying tabular data, with a boosting-compliant \textit{adversarial} training algorithm for GTs. We also introduce \textit{copycat training}, in which the generator copies at run time the underlying tree (graph) of the discriminator DT and completes it for the hardest discriminative task, with boosting compliant convergence. We test our algorithms on tasks including fake/real distinction and missing data imputation.

> Oral: Do More Negative Samples Necessarily Hurt In Contrastive Learning?
> 
> Authors: Pranjal Awasthi and Nishanth Dikkala and Pritish Kamath
> 
> Abstract: Recent investigations in noise contrastive estimation suggest, both empirically as well as theoretically, that while having more negative samples'' in the contrastive loss improves downstream classification performance initially, but beyond a threshold, it results in worse downstream classification performance due to acollision-coverage'' tradeoff. But is such a phenomenon inherent in contrastive learning?We show in a simple framework, where positive pairs are generated by sampling from the underlying latent class (introduced by Saunshi et al. (ICML 2019)), that the downstream performance of the representation optimizing the (population) contrastive loss in fact does not degrade with the number of negative samples. Along the way, we give a structural characterization of the optimal representation under such types of noise contrastive estimation. We also provide empirical support for our observations on CIFAR-10 and CIFAR-100 datasets.


> Oral: Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
> 
> Authors: Samuel Holt and Zhaozhi Qian and Mihaela van der Schaar
> 
> Abstract: Neural Ordinary Differential Equations model dynamical systems with \textit{ODE}s learned by neural networks.However, ODEs are fundamentally inadequate to model systems with long-range dependencies or discontinuities, which are common in engineering and biological systems. Broader classes of differential equations (DE) have been proposed as remedies, including delay differential equations and integro-differential equations.Furthermore, Neural ODE suffers from numerical instability when modelling stiff ODEs and ODEs with piecewise forcing functions.In this work, we propose \textit{Neural Laplace}, a unifying framework for learning diverse classes of DEs including all the aforementioned ones.Instead of modelling the dynamics in the time domain, we model it in the Laplace domain, where the history-dependencies and discontinuities in time can be represented as summations of complex exponentials. To make learning more efficient, we use the geometrical stereographic map of a Riemann sphere to induce more smoothness in the Laplace domain.In the experiments, Neural Laplace shows superior performance in modelling and extrapolating the trajectories of diverse classes of DEs, including the ones with complex history dependency and abrupt changes.


> Oral: Not All Poisons are Created Equal: Robust Training against Data Poisoning
> 
> Authors: Yu Yang and Tian Yu Liu and Baharan Mirzasoleiman
> 
> Abstract: Data poisoning causes misclassification of test time samples by injecting maliciously crafted samples in the training data.  Existing defenses are often effective only against a specific type of targeted attack,  significantly degrade the generalization performance, are prohibitive for standard deep learning pipelines. In this work, we propose an efficient defense mechanism that significantly reduces the success rate of various data poisoning attacks, and provides theoretical guarantees for the performance of the model. We make the following observations:  (i) targeted attacks add bounded perturbations to a randomly selected subset of training data to match the gradient of the target; (ii) under bounded perturbations, only a small number of poisons can be optimized to have a gradient that is close enough to that of the target and make the attack successful; (iii) such examples move away from their original class and get isolated in the gradient space. We show that training on large gradient clusters of each class can successfully eliminate the effective poisons, and guarantee similar training dynamics to that of training on the full data. Our extensive experiments show that our method significantly decreases the success rate of the state-of-the-art targeted attacks, including Gradient Matching and Bullseye Poly-tope, and easily scales to large datasets.


> Oral: Adversarially trained neural representations are already as robust as biological neural representations
> 
> Authors: Chong Guo and Michael Lee and Guillaume Leclerc and Joel Dapello and Yug Rao and Aleksander Madry and James DiCarlo
> 
> Abstract: Visual systems of primates are the gold standard of robust perception. There is thus a general belief that simply mimicking the neural structure that underlies such systems will yield artificial visual systems that are adversarially robust. In this work, we develop a method for performing adversarial visual attacks directly on primate brains. We then leverage this method to demonstrate that the above-mentioned belief might not be well founded. Specifically, we show that the biological neurons that make up visual systems of primates exhibit susceptibility to adversarial perturbations that is comparable to existing (robustly trained) artificial neural networks.


> Oral: Privacy for Free: How does Dataset Condensation Help Privacy?
> 
> Authors: Tian Dong and Bo Zhao and Lingjuan Lyu
> 
> Abstract: To prevent unintentional data leakage, research community has resorted to data generators that can produce differentially private data for model training. However, for the sake of the data privacy, existing solutions suffer from either expensive training cost or poor generalization performance. Therefore, we raise the question whether training efficiency and privacy can be achieved simultaneously. In this work, we for the first time identify that dataset condensation (DC) which is originally designed for improving training efficiency can be a better solution to replace data generators for private data generation, thus providing privacy for free. To demonstrate the privacy benefit of DC, we build a connection between DC and differential privacy (DP), and theoretically prove on linear feature extractors (and then extended to non-linear feature extractors) that the existence of one sample has limited impact (O(m/n)) on the parameter distribution of networks trained on m samples synthesized from n (n >> m) raw data by DC. We also empirically validate the vision privacy and membership privacy of DC-synthesized data by launching both the loss-based and the state-of-the-art likelihood-based membership inference attacks. We envision this work as a milestone for data-efficient and privacy-preserving machine learning.


> Oral: Toward Compositional Generalization in Object-Oriented World Modeling
> 
> Authors: Linfeng Zhao and Lingzhi Kong and Robin Walters and Lawson Wong
> 
> Abstract: Compositional generalization is a critical ability in learning and decision-making.We focus on the setting of reinforcement learning in object-oriented environments to study compositional generalization in world modeling.We (1) formalize the compositional generalization problem with an algebraic approach and (2) study how a world model can achieve that.We introduce a conceptual environment, Object Library, and two instances, and deploy a principled pipeline to measure the generalization ability.Motivated by the formulation, we analyze several methods with exact or no compositional generalization ability using our framework, and design a differentiable approach, Homomorphic Object-oriented World Model (HOWM), that achieves approximate but more efficient compositional generalization.


> Oral: Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints
> 
> Authors: Alina Ene and Huy Nguyen
> 
> Abstract: Maximizing a monotone k-submodular function subject to cardinality constraints is a general model for several applications ranging from influence maximization with multiple products to sensor placement with multiple sensor types and online ad allocation. Due to the large problem scale in many applications and the online nature of ad allocation, a need arises for algorithms that process elements in a streaming fashion and possibly make online decisions. In this work, we develop a new streaming algorithm for maximizing a monotone k-submodular function subject to a per-coordinate cardinality constraint attaining an approximation guarantee close to the state of the art guarantee in the offline setting. Though not typical for streaming algorithms, our streaming algorithm also readily applies to the online setting with free disposal. Our algorithm is combinatorial and enjoys fast running time and small number of function evaluations. Furthermore, its guarantee improves as the cardinality constraints get larger, which is especially suited for the large scale applications. For the special case of maximizing a submodular function with large budgets, our combinatorial algorithm matches the guarantee of the state-of-the-art continuous algorithm, which requires significantly more time and function evaluations.


> Oral: Monarch: Expressive Structured Matrices for Efficient and Accurate Training
> 
> Authors: Tri Dao and Beidi Chen and Nimit Sohoni and Arjun Desai and Michael Poli and Jessica Grogan and Alexander Liu and Aniruddh Rao and Atri Rudra and Christopher Re
> 
> Abstract: Large neural networks excel in many domains, but they are expensive to train and fine-tune. A popular approach to reduce their compute/memory requirements is to replace dense weight matrices with structured ones (e.g., sparse, low-rank, Fourier transform). These methods have not seen widespread adoption (1) in end-to-end training due to unfavorable efficiency--quality tradeoffs, and (2) in dense-to-sparse fine-tuning due to lack of tractable algorithms to approximate a given dense weight matrix. To address these issues, we propose a class of matrices (Monarch) that is \emph{hardware-efficient} (they are parameterized as products of two block-diagonal matrices for better hardware utilization) and \emph{expressive} (they can represent many commonly used transforms). Surprisingly, the problem of approximating a dense weight matrix with a Monarch matrix, though nonconvex, has an analytical optimal solution. These properties of Monarch matrices unlock new ways to train and fine-tune sparse and dense models. We empirically validate that Monarch can achieve favorable accuracy–efficiency tradeoffs in several end-to-end sparse training applications: speeding up ViT and GPT-2 training on ImageNet classification and Wikitext-103 language modeling by 2x with comparable model quality, and reducing the error on PDE solving and MRI reconstruction tasks by 40\%. In sparse-to-dense training, with a simple technique called ``reverse sparsification,'' Monarch matrices serve as a useful intermediate representation to speed up GPT-2 pretraining on OpenWebText by 2x without quality drop. In dense-to-sparse fine-tuning, as a proof-of-concept, our Monarch approximation algorithm speeds up BERT fine-tuning on GLUE by 1.7x with comparable accuracy.


> Oral: Tractable Uncertainty for Structure Learning
> 
> Authors: Benjie Wang and Matthew Wicker and Marta Kwiatkowska
> 
> Abstract: Bayesian structure learning allows one to capture uncertainty over the causal directed acyclic graph (DAG) responsible for generating given data. In this work, we present Tractable Uncertainty for STructure learning (TRUST), a framework for approximate posterior inference that relies on probabilistic circuits as a representation of our posterior belief. In contrast to sample-based posterior approximations, our representation can capture a much richer space of DAGs, while being able to tractably answer a range of useful inference queries. We empirically demonstrate how probabilistic circuits can be used to as an augmented representation for  structure learning methods, leading to improvement in both the quality of inferred structures and posterior uncertainty. Experimental results also demonstrate the improved representational capacity of TRUST, outperforming competing methods on conditional query answering.


> Oral: Contrastive Mixture of Posteriors for Counterfactual Inference, Data Integration and Fairness
> 
> Authors: Adam Foster and Arpi Vezer and Craig Glastonbury and Páidí Creed and Sam Abujudeh and Aaron Sim
> 
> Abstract: Learning meaningful representations of data that can address challenges such as batch effect correction and counterfactual inference is a central problem in many domains including computational biology. Adopting a Conditional VAE framework, we show that marginal independence between the representation and a condition variable plays a key role in both of these challenges. We propose the Contrastive Mixture of Posteriors (CoMP) method that uses a novel misalignment penalty defined in terms of mixtures of the variational posteriors to enforce this independence in latent space. We show that CoMP has attractive theoretical properties compared to previous approaches and we prove counterfactual identifiability of CoMP under additional assumptions. We demonstrate state of the art performance on a set of challenging tasks including aligning human tumour samples with cancer cell-lines, predicting transcriptome-level perturbation responses, and batch correction on single-cell RNA sequencing data. We also find parallels to fair representation learning and demonstrate that CoMP is competitive on a common task in the field.


> Oral: Causal Imitation Learning under Temporally Correlated Noise
> 
> Authors: Gokul Swamy and Sanjiban Choudhury and James Bagnell and Steven Wu
> 
> Abstract: We develop algorithms for imitation learning from policy data that was corrupted by temporally correlated noise in expert actions. When noise affects multiple timesteps of recorded data, it can manifest as spurious correlations between states and actions that a learner might latch on to, leading to poor policy performance. To break up these spurious correlations, we apply modern variants of the instrumental variable regression (IVR) technique of econometrics, enabling us to recover the underlying policy without requiring access to an interactive expert. In particular, we present two techniques, one of a generative-modeling flavor (DoubIL) that can utilize access to a simulator, and one of a game-theoretic flavor (ResiduIL) that can be run entirely offline. We find both of our algorithms compare favorably to behavioral cloning on simulated control tasks.


> Oral: Optimal Algorithms for Mean Estimation under Local Differential Privacy
> 
> Authors: Hilal Asi and Vitaly Feldman and Kunal Talwar
> 
> Abstract: We study the problem of mean estimation of $\ell_2$-bounded vectors under the constraint of local differential privacy. While the literature has a variety of algorithms that achieve the (asymptotic) optimal rates for this problem, the performance of these algorithms in practice can vary significantly due to varying (and often large) hidden constants. In this work, we investigate the question of designing the randomizer with the smallest variance. We show that PrivUnit (Bhowmick et al. 2018) with optimized parameters achieves the optimal variance among a large family of natural randomizers. To prove this result, we establish some properties of local randomizers, and use symmetrization arguments that allow us to write the optimal randomizer as the optimizer of a certain linear program. These structural results, which should extend to other problems, then allow us to show that the optimal randomizer belongs to the PrivUnit family.    We also develop a new variant of PrivUnit based on the Gaussian distribution which is more amenable to mathematical analysis and enjoys the same optimality guarantees. This allows us to establish several useful properties on the exact constants of the optimal error as well as to numerically estimate these constants.

> Oral: Learning Markov Games with Adversarial Opponents: Efficient Algorithms and Fundamental Limits
> 
> Authors: Qinghua Liu and Yuanhao Wang and Chi Jin
> 
> Abstract: An ideal strategy in zero-sum games should not only grant the player an average reward no less than the value of Nash equilibrium, but also exploit the (adaptive) opponents when they are suboptimal. While most existing works in Markov games focus exclusively on the former objective, it remains open whether we can achieve both objectives simultaneously. To address this problem, this work studies no-regret learning in Markov games with adversarial opponents when competing against the best fixed policy in hindsight. Along this direction, we present a new complete set of positive and negative results:When the policies of the opponents are revealed at the end of each episode, we propose new efficient algorithms achieving $\sqrt{K}$ regret bounds when either (1) the baseline policy class is small or (2) the opponent’s policy class is small. This is complemented with an exponential lower bound when neither conditions are true. When the policies of the opponents are not revealed, we prove a statistical hardness result even in the most favorable scenario when both above conditions are true. Our hardness result is much stronger than the existing hardness results which either only involve computational hardness, or require further restrictions on the algorithms.

> Oral: To Smooth or Not? When Label Smoothing Meets Noisy Labels
> 
> Authors: Jiaheng Wei and Hangyu Liu and Tongliang Liu and Gang Niu and Masashi Sugiyama and Yang Liu
> 
> Abstract: Label smoothing (LS) is an arising learning paradigm that uses the positively weighted average of both the hard training labels and uniformly distributed soft labels. It was shown that LS serves as a regularizer for training data with hard labels and therefore improves the generalization of the model. Later it was reported LS even helps with improving robustness when learning with noisy labels. However, we observed that the advantage of LS vanishes when we operate in a high label noise regime. Intuitively speaking, this is due to the increased entropy of P(noisy label|X) when the noise rate is high, in which case, further applying LS tends to “over-smooth” the estimated posterior. We proceeded to discover that several learning-with-noisy-labels solutions in the literature instead relate more closely to not/negative label smoothing (NLS), which acts counter to LS and defines as using a negative weight to combine the hard and soft labels! We provide understandings for the properties of LS and NLS when learning with noisy labels. Among other established properties, we theoretically show NLS is considered more beneficial when the label noise rates are high. We provide extensive experimental results on multiple benchmarks to support our findings too.


> Oral: Bounding Training Data Reconstruction in Private (Deep) Learning
> 
> Authors: Chuan Guo and Brian Karrer and Kamalika Chaudhuri and Laurens van der Maaten
> 
> Abstract: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods---Renyi differential privacy and Fisher information leakage---both offer strong semantic protection against data reconstruction attacks.


> Oral: Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum
> 
> Authors: Zeke Xie and Xinrui Wang and Huishuai Zhang and Issei Sato and Masashi Sugiyama
> 
> Abstract: Adaptive Momentum Estimation (Adam), which combines Adaptive Learning Rate and Momentum, would be the most popular stochastic optimizer for accelerating the training of deep neural networks. However, it is empirically known that Adam often generalizes worse than Stochastic Gradient Descent (SGD). The purpose of this paper is to unveil the mystery of this behavior in the diffusion theoretical framework. Specifically, we disentangle the effects of Adaptive Learning Rate and Momentum of the Adam dynamics on saddle-point escaping and flat minima selection. We prove that Adaptive Learning Rate can escape saddle points efficiently, but cannot select flat minima as SGD does. In contrast, Momentum provides a drift effect to help the training process pass through saddle points, and almost does not affect flat minima selection. This partly explains why SGD (with Momentum) generalizes better, while Adam generalizes worse but converges faster. Furthermore, motivated by the analysis, we design a novel adaptive optimization framework named Adaptive Inertia, which uses parameter-wise adaptive inertia to accelerate the training and provably favors flat minima as well as SGD. Our extensive experiments demonstrate that the proposed adaptive inertia method can generalize significantly better than SGD and conventional adaptive gradient methods.


> Oral: Offline RL Policies Should Be Trained to be Adaptive
> 
> Authors: Dibya Ghosh and Anurag Ajay and Pulkit Agrawal and Sergey Levine
> 
> Abstract: Offline RL algorithms must learn policies that achieve high return, given only a fixed dataset of experience from an environment. Successfully doing so requires grappling with partial specification, in that the limited set of transitions leaves facets of the environment unknown. The most common way to approach this challenge is to employ pessimistic or conservative methods, which avoid behaviors that are too dissimilar from those in the training dataset. However, relying exclusively on conservatism has drawbacks: performance is sensitive to the exact degree of conservatism, and conservative objectives can recover highly suboptimal policies.In this work, we propose that offline RL methods should instead be adaptive in the presence of uncertainty. This requires learning policies that at test-time condition their behavior on all the transitions seen so far during evaluation, not only the current state. We show that adaptive policies are optimal for offline RL in a Bayesian sense, and discuss exactly how policies must adapt to be optimal. We present a model-free algorithm for approximating this optimal adaptive policy, and demonstrate the efficacy of learning policies with this adaptation mechanism in several offline RL benchmarks.


> Oral: Continuous-Time Analysis of Accelerated Gradient Methods via Conservation Laws in Dilated Coordinate Systems
> 
> Authors: Jaewook Suh and Gyumin Roh and Ernest Ryu
> 
> Abstract: We analyze continuous-time models of accelerated gradient methods through deriving conservation laws in dilated coordinate systems. Namely, instead of analyzing the dynamics of $X(t)$, we analyze the dynamics of $W(t)=t^\alpha(X(t)-X_c)$ for some $\alpha$ and $X_c$ and derive a conserved quantity, analogous to physical energy, in this dilated coordinate system. Through this methodology, we recover many known continuous-time analyses in a streamlined manner and obtain novel continuous-time analyses for OGM-G, an acceleration mechanism for efficiently reducing gradient magnitude that is distinct from that of Nesterov. Finally, we show that a semi-second-order symplectic Euler discretization in the dilated coordinate system leads to an $\mathcal{O}(1/k^2)$ rate on the standard setup of smooth convex minimization, without any further assumptions such as infinite differentiability.

> Oral: The Poisson Binomial Mechanism for Unbiased Federated Learning with Secure Aggregation
> 
> Authors:   and Ayfer Ozgur and Peter Kairouz
> 
> Abstract: We introduce the Poisson Binomial mechanism (PBM), a discrete differential privacy mechanism for distributed mean estimation (DME) with applications to federated learning and analytics. We provide a tight analysis of its privacy guarantees, showing that it achieves the same privacy-accuracy trade-offs as the continuous Gaussian mechanism. Our analysis is based on a novel bound on the R\'enyi divergence of two Poisson binomial distributions that may be of independent interest. Unlike previous discrete DP schemes based on additive noise, our mechanism encodes local information into a parameter of the binomial distribution, and hence the output distribution is discrete with bounded support. Moreover, the support does not increase as the privacy budget goes to zero as in the case of additive schemes which require the addition of more noise to achieve higher privacy; on the contrary, the support becomes smaller as eps goes to zero. The bounded support enables us to combine our mechanism with secure aggregation (SecAgg), a multi-party cryptographic protocol,  without the need of performing modular clipping which results in an unbiased estimator of the sum of the local vectors. This in turn allows us to  apply it in the private FL setting and provide an upper bound on the convergence rate of the  SGD algorithm. Moreover, since the support of the output distribution becomes smaller as $\varepsilon \ra 0$, the communication cost of our scheme decreases with the privacy constraint $\varepsilon$, outperforming all previous distributed DP schemes based on additive noise in the high privacy or low communication regimes. 

> Oral: Preconditioning for Scalable Gaussian Process Hyperparameter Optimization
> 
> Authors: Jonathan Wenger and Geoff Pleiss and Philipp Hennig and John Cunningham and Jacob Gardner
> 
> Abstract: Gaussian process hyperparameter optimization requires linear solves with, and log-determinants of, large kernel matrices. Iterative numerical techniques are becoming popular to scale to larger datasets, relying on the conjugate gradient method (CG) for the linear solves and stochastic trace estimation for the log-determinant. This work introduces new algorithmic and theoretical insights for preconditioning these computations. While preconditioning is well understood in the context of CG, we demonstrate that it can also accelerate convergence and reduce variance of the estimates for the log-determinant and its derivative. We prove general probabilistic error bounds for the preconditioned computation of the log-determinant, log-marginal likelihood and its derivatives. Additionally, we derive specific rates for a range of kernel-preconditioner combinations, showing that up to exponential convergence can be achieved. Our theoretical results enable provably efficient optimization of kernel hyperparameters, which we validate empirically on large-scale benchmark problems. There our approach accelerates training by up to an order of magnitude.


> Oral: Anarchic Federated Learning
> 
> Authors: Haibo Yang and Xin Zhang and Prashant Khanduri and Jia Liu
> 
> Abstract: Present-day federated learning (FL) systems deployed over edge networks consists of a large number of workers with high degrees of heterogeneity in data and/or computing capabilities, which call for flexible worker participation in terms of timing, effort, data heterogeneity, etc. To satisfy the need for flexible worker participation, we consider a new FL paradigm called ``Anarchic Federated Learning'' (AFL) in this paper. In stark contrast to conventional FL models, each worker in AFL has the freedom to choose i) when to participate in FL, and ii) the number of local steps to perform in each round based on its current situation (e.g., battery level, communication channels, privacy concerns). However, such chaotic worker behaviors in AFL impose many new open questions in algorithm design. In particular, it remains unclear whether one could develop convergent AFL training algorithms, and if yes, under what conditions and how fast the achievable convergence speed is. Toward this end, we propose two Anarchic Federated Averaging (AFA) algorithms with two-sided learning rates for both cross-device and cross-silo settings, which are named AFA-CD and AFA-CS, respectively.  Somewhat surprisingly, we show that, under mild anarchic assumptions, both AFL algorithms achieve the best known convergence rate as the state-of-the-art algorithms for conventional FL. Moreover, they retain the highly desirable {\em linear speedup effect} with respect of both the number of workers and local steps in the new AFL paradigm. We validate the proposed algorithms with extensive experiments on real-world datasets.


> Oral: Robustness Verification for Contrastive Learning
> 
> Authors: Zekai Wang and Weiwei Liu
> 
> Abstract: Contrastive adversarial training has successfully improved the robustness of contrastive learning (CL). However, the robustness metric used in these methods is linked to attack algorithms, image labels and downstream tasks, all of which may affect the consistency and reliability of robustness metric for CL. To address these problems, this paper proposes a novel Robustness Verification framework for Contrastive Learning (RVCL). Furthermore, we use extreme value theory to reveal the relationship between the robust radius of the CL encoder and that of the supervised downstream task. Extensive experimental results on various benchmark models and datasets verify our theoretical findings, and further demonstrate that our proposed RVCL is able to evaluate the robustness of both models and images.


> Oral: Exact Optimal Accelerated Complexity for Fixed-Point Iterations
> 
> Authors: Jisun Park and Ernest Ryu
> 
> Abstract: Despite the broad use of fixed-point iterations throughout applied mathematics, the optimal convergence rate of general fixed-point problems with nonexpansive nonlinear operators has not been established. This work presents an acceleration mechanism for fixed-point iterations with nonexpansive operators, contractive operators, and nonexpansive operators satisfying a H\"older-type growth condition. We then provide matching complexity lower bounds to establish the exact optimality of the acceleration mechanisms in the nonexpansive and contractive setups. Finally, we provide experiments with CT imaging, optimal transport, and decentralized optimization to demonstrate the practical effectiveness of the acceleration mechanism.


> Oral: H-Consistency Estimation Error of Surrogate Loss Minimizers
> 
> Authors: Pranjal Awasthi and Anqi Mao and Mehryar Mohri and Yutao Zhong
> 
> Abstract: We present a detailed study of estimation errors in terms of surrogate loss estimation errors.  We refer to such guarantees as H-consistency estimation error bounds, sincethey account for the hypothesis set H adopted. These guarantees are significantly stronger than H-calibration or H-consistency. They are also more informative than similar excess error bounds derived in the literature, when H is the family of all measurable functions. We prove general theorems providing such guarantees, for both the distribution-dependent and distribution-independent settings. We show that our bounds are tight, modulo a convexity assumption. We also show that previous excess error bounds can be recovered as special cases of our general results.We then present a series of explicit bounds in the case of the zero-one loss, with multiple choices of the surrogate loss and for both the family of linear functions and neural networks with one hidden-layer. We further prove more favorable distribution-dependent guarantees in that case. We also present a series of explicit bounds in the case of the adversarial loss, with surrogate losses based on the supremum of the $\rho$-margin, hinge or sigmoid loss and for the same two general hypothesis sets. Here too, we prove several enhancements of these guarantees under natural distributional assumptions.  Finally, we report the results of simulations illustrating our bounds and their tightness.

> Oral: Nonparametric Involutive Markov Chain Monte Carlo
> 
> Authors: Carol Mak and Fabian Zaiser and Luke Ong
> 
> Abstract: A challenging problem in probabilistic programming is to develop inference algorithms that work for arbitrary programs in a universal probabilistic programming language (PPL). We present the nonparametric involutive Markov chain Monte Carlo (NP-iMCMC) algorithm as a method for constructing MCMC inference algorithms for nonparametric models expressible in universal PPLs. Building on the unifying involutive MCMC framework, and by providing a general procedure for driving state movement between dimensions, we show that NP-iMCMC can generalise numerous existing iMCMC algorithms to work on nonparametric models. We prove the correctness of the NP-iMCMC sampler. Our empirical study shows that the existing strengths of several iMCMC algorithms carry over to their nonparametric extensions. Applying our method to the recently proposed Nonparametric HMC (an instance of NP-iMCMC), we have constructed several nonparametric extensions (all of which new) that exhibit significant performance improvements.


> Oral: Last Iterate Risk Bounds of SGD with Decaying Stepsize for Overparameterized Linear Regression
> 
> Authors: Jingfeng Wu and Difan Zou and Vladimir Braverman and Quanquan Gu and Sham Kakade
> 
> Abstract: Stochastic gradient descent (SGD) has been shown to generalize well in many deep learning applications. In practice, one often runs SGD with a geometrically decaying stepsize, i.e., a constant initial stepsize followed by multiple geometric stepsize decay, and uses the last iterate as the output. This kind of SGD is known to be nearly minimax optimal for classical finite-dimensional linear regression problems (Ge et al., 2019). However, a sharp analysis for the last iterate of SGD in the overparameterized setting is still open. In this paper, we provide a problem-dependent analysis on the last iterate risk bounds of SGD with decaying stepsize, for (overparameterized) linear regression problems. In particular, for last iterate SGD with (tail) geometrically decaying stepsize, we prove nearly matching upper and lower bounds on the excess risk. Moreover, we provide an excess risk lower bound for last iterate SGD with polynomially decaying stepsize and demonstrate the advantage of geometrically decaying stepsize in an instance-wise manner, which complements the minimax rate comparison made in prior work.


> Oral: Towards Noise-adaptive, Problem-adaptive (Accelerated) Stochastic Gradient Descent
> 
> Authors: Sharan Vaswani and Benjamin Dubois-Taine and Reza Babanezhad
> 
> Abstract: We aim to make stochastic gradient descent (SGD) adaptive to (i) the noise $\sigma^2$ in the stochastic gradients and (ii) problem-dependent constants. When minimizing smooth, strongly-convex functions with condition number $\kappa$, we prove that $T$ iterations of SGD with exponentially decreasing step-sizes and knowledge of the smoothness can achieve an $\tilde{O} \left(\exp \left( \nicefrac{-T}{\kappa} \right) + \nicefrac{\sigma^2}{T} \right)$ rate, without knowing $\sigma^2$. In order to be adaptive to the smoothness, we use a stochastic line-search (SLS) and show (via upper and lower-bounds) that SGD with SLS converges at the desired rate, but only to a neighbourhood of the solution. On the other hand, we prove that SGD with an offline estimate of the smoothness converges to the minimizer. However, its rate is slowed down proportional to the estimation error. Next, we prove that SGD with Nesterov acceleration and exponential step-sizes (referred to as ASGD) can achieve the near-optimal $\tilde{O} \left(\exp \left( \nicefrac{-T}{\sqrt{\kappa}} \right) + \nicefrac{\sigma^2}{T} \right)$ rate, without knowledge of $\sigma^2$. When used with offline estimates of the smoothness and strong-convexity, ASGD still converges to the solution, albeit at a slower rate. Finally, we empirically demonstrate the effectiveness of exponential step-sizes coupled with a novel variant of SLS.

> Oral: Improved No-Regret Algorithms for Stochastic Shortest Path with Linear MDP
> 
> Authors: Liyu Chen and Rahul Jain and Haipeng Luo
> 
> Abstract: We introduce two new no-regret algorithms for the stochastic shortest path (SSP) problem with a linear MDP that significantly improve over the only existing results of (Vial et al., 2021).Our first algorithm is computationally efficient and achieves a regret bound $O(\sqrt{d^3\B^2\T K})$, where $d$ is the dimension of the feature space, $\B$ and $\T$ are upper bounds of the expected costs and hitting time of the optimal policy respectively, and $K$ is the number of episodes.The same algorithm with a slight modification also achieves logarithmic regret of order $O(\frac{d^3\B^4}{\cmin^2\mingap}\ln^5\frac{d\B K}{\cmin})$, where $\mingap$ is the minimum sub-optimality gap and $\cmin$ is the minimum cost over all state-action pairs.Our result is obtained by developing a simpler and improved analysis for the finite-horizon approximation of (Cohen et al., 2021) with a smaller approximation error, which might be of independent interest.On the other hand, using variance-aware confidence sets in a global optimization problem,our second algorithm is computationally inefficient but achieves the first ``horizon-free'' regret bound $O(d^{3.5}\B\sqrt{K})$ with no polynomial dependency on $\T$ or $1/\cmin$,almost matching the $\Omega(d\B\sqrt{K})$ lower bound from (Min et al., 2021).

> Oral: Understanding Dataset Difficulty in NLP with $\mathcal{V}$-Usable Information
> 
> Authors: Kawin Ethayarajh and Yejin Choi and Swabha Swayamdipta
> 
> Abstract: Estimating the difficulty of an NLP dataset typically involves comparing state-of-the-art models to humans; the bigger the performance gap, the harder the dataset is said to be. However, this comparison provides little understanding of how difficult each instance is in a given distribution, or what attributes make the dataset difficult for a given model. To address these questions, we frame dataset difficulty---w.r.t. a model $\mathcal{V}$---as the lack of $\mathcal{V}$-usable information (Xu et al., 2019), where a lower value indicates a more difficult dataset for $\mathcal{V}$. We further introduce pointwise $\mathcal{V}$-information (PVI) for measuring the difficulty of individual instances w.r.t. a given distribution. While standard evaluation metrics typically only compare different models for the same dataset, $\mathcal{V}$-usable information and PVI also permit the converse: for a given model $\mathcal{V}$, we can compare different datasets, as well as different instances/slices of the same dataset. Furthermore, our framework allows for the interpretability of different input attributes via transformations of the input, which we use to discover annotation artefacts in widely-used NLP benchmarks. 

> Oral: BAMDT: Bayesian Additive Partial Multivariate Decision Trees for Nonparametric Regression
> 
> Authors: Zhao Tang Luo and Huiyan Sang and Bani Mallick
> 
> Abstract: Bayesian additive regression trees (BART; Chipman et al., 2010) have gained great popularity as a flexible nonparametric function estimation and modeling tool. Nearly all existing BART models rely on decision tree weak learners with univariate split rules to partition the Euclidean feature space into rectangular regions. In practice, however, many regression problems  involve features with multivariate structures (e.g., spatial locations) possibly lying in a manifold, where rectangular partitions may fail to respect irregular intrinsic geometry and boundary constraints of the structured feature space. In this paper, we develop a new class of Bayesian additive multivariate decision tree models that combine univariate split rules for handling possibly high dimensional features without known multivariate structures and novel multivariate split rules for features with multivariate structures in each weak learner. The proposed multivariate split rules are built upon stochastic predictive spanning tree bipartition models on reference knots, which are capable of achieving highly flexible nonlinear decision boundaries on manifold feature spaces while enabling efficient dimension reduction computations.  We demonstrate the superior performance of the proposed method over BART and Gaussian process regression models using simulation data and a Sacramento housing price data set.


> Oral: Cooperative Online Learning in Stochastic and Adversarial MDPs
> 
> Authors: Tal Lancewicki and Aviv Rosenberg and Yishay Mansour
> 
> Abstract:  We study cooperative online learning in stochastic and adversarial Markov decision process (MDP). That is, in each episode, $m$ agents interact with an MDP simultaneously and share information in order to minimize their individual regret. We consider environments with two types of randomness: \emph{fresh} -- where each agent's trajectory is sampled i.i.d, and \emph{non-fresh} -- where the realization is shared by all agents (but each  agent's trajectory is also affected by its own actions). More precisely, with non-fresh randomness the realization of every cost and transition is fixed at the start of each episode, and agents that take the same action in the same state at the same time observe the same cost and next state. We thoroughly analyze all relevant settings, highlight the challenges and differences between the models, and prove nearly-matching regret lower and upper bounds. To our knowledge, we are the first to consider cooperative reinforcement learning (RL) with either non-fresh randomness or in adversarial MDPs.

> Oral: A General Recipe for Likelihood-free Bayesian Optimization
> 
> Authors: Jiaming Song and Lantao Yu and Willie Neiswanger and Stefano Ermon
> 
> Abstract: The acquisition function, a critical component in Bayesian optimization (BO), can often be written as the expectation of a utility function under a surrogate model. However, to ensure that acquisition functions are tractable to optimize, restrictions must be placed on the surrogate model and utility function. To extend BO to a broader class of models and utilities, we propose likelihood-free BO (LFBO), an approach based on likelihood-free inference. LFBO directly models the acquisition function without having to separately perform inference with a probabilistic surrogate model. We show that computing the acquisition function in LFBO can be reduced to optimizing a weighted classification problem, which extends an existing likelihood-free density ratio estimation method related to probability of improvement (PI). By choosing the utility function for expected improvement (EI), LFBO outperforms the aforementioned method, as well as various state-of-the-art black-box optimization methods on several real-world optimization problems. LFBO can also leverage composite structures of the objective function, which further improves its regret by several orders of magnitude.


> Oral: Agnostic Learnability of Halfspaces via Logistic Loss
> 
> Authors: Ziwei Ji and Kwangjun Ahn and Pranjal Awasthi and Satyen Kale and Stefani Karp
> 
> Abstract: We investigate approximation guarantees provided by logistic regression for the fundamental problem of agnostic learning of homogeneous halfspaces. Previously, for a certain broad class of “well-behaved” distributions on the examples, Diakonikolas et al. (2020) proved an tilde{Omega}(OPT) lower bound, while Frei et al. (2021) proved an tilde{O}(sqrt{OPT}) upper bound, where OPT denotes the best zero-one/misclassification risk of a homogeneous halfspace. In this paper, we close this gap by constructing a well-behaved distribution such that the global minimizer of the logistic risk over this distribution only achieves Omega(sqrt{OPT}) misclassification risk, matching the upper bound in (Frei et al., 2021). On the other hand, we also show that if we impose a radial-Lipschitzness condition in addition to well-behaved-ness on the distribution, logistic regression on a ball of bounded radius reaches tilde{O}(OPT) misclassification risk. Our techniques also show for any well-behaved distribution, regardless of radial Lipschitzness, we can overcome the Omega(sqrt{OPT}) lower bound for logistic loss simply at the cost of one additional convex optimization step involving the hinge loss and attain tilde{O}(OPT) misclassification risk. This two-step convex optimization algorithm is simpler than previous methods obtaining this guarantee, all of which require solving O(log(1/OPT)) minimization problems.


> Oral: Learning inverse folding from millions of predicted structures
> 
> Authors: Chloe Hsu and Robert Verkuil and Jason Liu and Zeming Lin and Brian Hie and Tom Sercu and Adam Lerer and Alexander Rives
> 
> Abstract: We consider the problem of predicting a protein sequence from its backbone atom coordinates. Machine learning approaches to this problem to date have been limited by the number of experimentally determined protein structures available for learning. We augment training data by two orders of magnitude by predicting structures for 12M protein sequences using AlphaFold2. Trained with this additional data, a sequence-to-sequence transformer with invariant geometric input processing layers achieves 51% native sequence recovery on structurally held-out backbones with 71% recovery for buried residues, an overall improvement of almost 10 percentage points over existing methods. The model generalizes to a variety of more complex tasks including design of protein complexes, partially masked structures, binding interfaces, and multiple states.


> Oral: It’s Raw! Audio Generation with State-Space Models
> 
> Authors: Karan Goel and Albert Gu and Chris Donahue and Christopher Re
> 
> Abstract: Developing architectures suitable for modeling raw audio is a challenging problem due to the high sampling rates of audio waveforms.  Standard sequence modeling approaches like RNNs and CNNs have previously been tailored to fit the demands of audio, but the resultant architectures make undesirable computational tradeoffs and struggle to model waveforms effectively.  We propose Sashimi, a new multi-scale architecture for waveform modeling built around the recently introduced S4 model for long sequence modeling.  We identify that S4 can be unstable during autoregressive generation, and provide a simple improvement to its parameterization drawing connections to Hurwitz matrices.  Sashimi yields state-of-the-art performance for unconditional waveform generation in the autoregressive setting.  Additionally, Sashimi improves non-autoregressive generation performance when used as the backbone architecture for a diffusion model.  Compared to prior architectures in the autoregressive generation setting, Sashimi generates piano and speech waveforms which humans find more musical and coherent respectively, e.g. 2X better mean opinion scores than WaveNet on an unconditional speech generation task.  On a music generation task, Sashimi outperforms WaveNet on density estimation and speed at both training and inference even when using 3X fewer parameters. 


> Oral: RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests
> 
> Authors: Victor Chernozhukov and Whitney Newey and Víctor Quintas-Martínez and Vasilis Syrgkanis
> 
> Abstract: Many causal and policy effects of interest are defined by linear functionals of high-dimensional or non-parametric regression functions. $\sqrt{n}$-consistent and asymptotically normal estimation of the object of interest requires debiasing to reduce the effects of regularization and/or model selection on the object of interest. Debiasing is typically achieved by adding a correction term to the plug-in estimator of the functional, that is derived based on a functional-specific theoretical derivation of what is known as the influence function and which leads to properties such as double robustness and Neyman orthogonality. We instead implement an automatic debiasing procedure based on automatically learning the Riesz representation of the linear functional using Neural Nets and Random Forests. Our method solely requires value query oracle access to the linear functional. We propose a multi-tasking Neural Net debiasing method with stochastic gradient descent minimization of a combined Riesz representer and regression loss, while sharing representation layers for the two functions. We also propose a Random Forest method which learns a locally linear representation of the Riesz function. Even though our methodology applies to arbitrary functionals, we experimentally find that it beats state of the art performance of the prior neural net based estimator of Shi et al. (2019) for the case of the average treatment effect functional. We also evaluate our method on the more challenging problem of estimating average marginal effects with continuous treatments, using semi-synthetic data of gasoline price changes on gasoline demand.

> Oral: Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning
> 
> Authors: Sixing Yu and Ali Jannesari and Arya Mazaheri
> 
> Abstract: Model compression is an essential technique for deploying deep neural networks (DNNs) on power and memory-constrained resources. However,  existing model-compression methods often rely on human expertise and focus on parameters' local importance, ignoring the rich topology information within DNNs. In this paper, we propose a novel multi-stage graph embedding technique based on graph neural networks (GNNs) to identify DNN topologies and use reinforcement learning (RL) to find a suitable compression policy. We performed resource-constrained (i.e., FLOPs) channel pruning and compared our approach with state-of-the-art model compression methods.We evaluated our method on various models from typical to mobile-friendly networks, such as ResNet family, VGG-16, MobileNet-v1/v2, and ShuffleNet. Results show that with minimal fine-tuning, our method can achieve higher compression ratios with outstanding and competitive performance.


> Oral: Generalized Results for the Existence and Consistency of the MLE in the Bradley-Terry-Luce Model
> 
> Authors: Heejong Bong and Alessandro Rinaldo
> 
> Abstract: Ranking problems based on pairwise comparisons, such as those arising in online gaming, often involve a large pool of items to order.  In these situations, the gap in performance between any two items can be significant, and the smallest and largest winning probabilities can be very close to zero or one. Furthermore, each item may be compared only to a subset of all the items, so that not all pairwise comparisons are observed. In this paper, we study the performance of the Bradley-Terry-Luce model for ranking from pairwise comparison data under more realistic settings than those considered in the literature so far. In particular, we allow for near-degenerate winning probabilities and arbitrary comparison designs. We obtain novel results about the existence of the maximum likelihood estimator (MLE) and the corresponding $\ell_2$ estimation error without the bounded winning probability assumption commonly used in the literature and for arbitrary comparison graph topologies. Central to our approach is the reliance on the Fisher information matrix to express the dependence on the graph topologies and the impact of the values of the winning probabilities on the estimation risk and on the conditions for the existence of the MLE. Our bounds recover existing results as special cases but are more broadly applicable.

> Oral: A Convergent and Dimension-Independent Min-Max Optimization Algorithm
> 
> Authors: Vijay Keswani and Oren Mangoubi and Sushant Sachdeva and Nisheeth K. Vishnoi
> 
> Abstract: We study a variant of a recently introduced min-max optimization framework where the max-player is constrained to update its parameters in a greedy manner until it reaches a first-order stationary point. Our equilibrium definition for this framework depends on a proposal distribution which the min-player uses to choose directions in which to update its parameters. We show that, given a smooth and bounded nonconvex-nonconcave objective function, access to any proposal distribution for the min-player’s updates, and stochastic gradient oracle for the max-player, our algorithm converges to the aforementioned approximate local equilibrium in a number of iterations that does not depend on the dimension. The equilibrium point found by our algorithm depends on the proposal distribution, and when applying our algorithm to train GANs we choose the proposal distribution to be a distribution of stochastic gradients.  We empirically evaluate our algorithm on challenging nonconvex-nonconcave test-functions and loss functions arising in GAN training. Our algorithm converges on these test functions and, when used to train GANs, trains stably on synthetic and real-world datasets and avoids mode collapse.


> Oral: Label Ranking through Nonparametric Regression
> 
> Authors: Dimitris Fotakis and Alkis Kalavasis and Eleni Psaroudaki
> 
> Abstract: Label Ranking (LR) corresponds to the problem of learning a hypothesis that maps features to rankings over a finite set of labels. We adopt a nonparametric regression approach to LR and obtain theoretical performance guarantees for this fundamental practical problem. We introduce a generative model for Label Ranking, in noiseless and noisy nonparametric regression settings, and provide sample complexity bounds for learning algorithms in both cases. In the noiseless setting, we study the  LR problem with full rankings and provide computationally efficient algorithms using decision trees and random forests in the high-dimensional regime. In the noisy setting, we consider the more general cases of LR with incomplete and partial rankings from a statistical viewpoint and obtain sample complexity bounds using the One-Versus-One approach of multiclass classification. Finally, we complement our theoretical contributions with experiments, aiming to understand how the input regression noise affects the observed output.


> Oral: The Unsurprising Effectiveness of Pre-Trained Vision Models for Control
> 
> Authors: Simone Parisi and Aravind Rajeswaran and Senthil Purushwalkam and Abhinav Gupta
> 
> Abstract: Recent years have seen the emergence of pre-trained representations as a powerful abstraction for downstream AI applications in computer vision, natural language, and speech. However, policy learning for control is still dominated by a tabula-rasa learning paradigm where visuo-motor policies are often learned from scratch with data from the deployment environments. In this context, we revisit the role of off-the-shelf pre-trained visual representations for control, carefully studying the importance of representation training methods, data augmentations, and feature hierarchies. Through extensive evaluations in four commonly studied domains --Habitat, DeepMind Control, Adroit, and Franka Kitchen-- we find that, when used correctly, pre-trained visual representations can be competitive or even better than ground-truth states for train control policies. This is inspite of using only out-domain data from standard vision datasets, without any in-domain data from the deployment environments.


> Oral: Online Active Regression
> 
> Authors: Cheng Chen and Yi Li and Yiming Sun
> 
> Abstract: Active regression considers a linear regression problem where the learner receives a large number of data points but can only observe a small number of labels. Since online algorithms can deal with incremental training data and take advantage of low computational cost, we consider an online extension of the active regression problem: the learner receives data points one by one and immediately decides whether it should collect the corresponding labels. The goal is to efficiently maintain the regression of received data points with a small budget of label queries. We propose novel algorithms for this problem under $\ell_p$ loss where $p\in[1,2]$. To achieve a $(1+\epsilon)$-approximate solution, our proposed algorithms only requires $\tilde{\mathcal{O}}(d/poly(\epsilon))$ queries of labels. The numerical results verify our theoretical results and show that our methods have comparable performance with offline active regression algorithms.

> Oral: Score matching enables causal discovery of nonlinear additive noise models
> 
> Authors: Paul Rolland and Volkan Cevher and Matthäus Kleindessner and Chris Russell and Dominik Janzing and Bernhard Schölkopf and Francesco Locatello
> 
> Abstract: This paper demonstrates how to recover causal graphs from the score of the data distribution in non-linear additive (Gaussian) noise models. Using score matching algorithms as a building block, we show how to design a new generation of scalable causal discovery methods. To showcase our approach, we also propose a new efficient method for approximating the score's Jacobian, enabling to recover the causal graph. Empirically, we find that the new algorithm, called SCORE, is competitive with state-of-the-art causal discovery methods while being significantly faster.


> Oral: First-Order Regret in Reinforcement Learning with Linear Function Approximation: A Robust Estimation Approach
> 
> Authors: Andrew Wagenmaker and Yifang Chen and Max Simchowitz and Simon Du and Kevin Jamieson
> 
> Abstract: Obtaining first-order regret bounds---regret bounds scaling not as the worst-case but with some measure of the performance of the optimal policy on a given instance---is a core question in sequential decision-making. While such bounds exist in many settings, they have proven elusive in reinforcement learning with large state spaces. In this work we address this gap, and show that it is possible to obtain regret scaling as $\mathcal{O}(\sqrt{V_1^\star K})$ in reinforcement learning with large state spaces, namely the linear MDP setting. Here  $V_1^\star$ is the value of the optimal policy and $K$ is the number of episodes. We demonstrate that existing techniques based on least squares estimation are insufficient to obtain this result, and instead develop a novel robust self-normalized concentration bound based on the robust Catoni mean estimator, which may be of independent interest.

> Oral: Born-Infeld (BI) for AI:  Energy-Conserving Descent (ECD) for Optimization
> 
> Authors: Giuseppe Bruno De Luca and Eva Silverstein
> 
> Abstract: We introduce a novel framework for optimization based on energy-conserving Hamiltonian dynamics in a strongly mixing (chaotic) regime and establish its key properties analytically and numerically.  The prototype is a discretization of Born-Infeld dynamics, with a squared relativistic speed limit depending on the objective function. This class of frictionless, energy-conserving optimizers proceeds unobstructed until slowing naturally near the minimal loss, which dominates the phase space volume of the system.  Building from studies of chaotic systems such as dynamical billiards, we formulate a specific algorithm with good performance on machine learning and PDE-solving tasks, including generalization. It cannot stop at a high local minimum and cannot overshoot the global minimum, yielding an advantage in non-convex loss functions, and proceeds faster than GD+momentum in shallow valleys.  


> Oral: Solving Stackelberg Prediction Game with Least Squares Loss Via Spherically Constrained Least Squares Reformulation
> 
> Authors: jiali wang and Wen Huang and Rujun Jiang and Xudong Li and Alex Wang
> 
> Abstract: The Stackelberg prediction game (SPG) is popular in characterizing strategic interactions between a learner and an attacker. As an important special case, the SPG with least squares loss (SPG-LS) has recently received much research attention. Although initially formulated as a difficult bi-level optimization problem, SPG-LS admits tractable reformulations which can be polynomially globally solved by semidefinite programming or second order cone programming. However, all the available approaches are not well-suited for handling large-scale datasets, especially those with huge numbers of features.  In this paper, we explore an alternative reformulation of the SPG-LS. By a novel nonlinear change of variables, we rewrite the SPG-LS  as a spherically constrained least squares (SCLS) problem. Theoretically, we show that an $\epsilon$ optimal solutions to the SCLS (and the SPG-LS) can be achieved in $\tilde O(N/\sqrt{\epsilon})$ floating-point operations, where $N$ is the number of nonzero entries in the data matrix. Practically, we apply two well-known methods for solving this new reformulation, i.e., the Krylov subspace method and the Riemannian trust region method. Both algorithms are factorization free so that they are suitable for solving large scale problems. Numerical results on both synthetic and real-world datasets indicate that the SPG-LS, equipped with the SCLS reformulation, can be solved orders of magnitude faster than the state of the art.

> Oral: From Dirichlet to Rubin: Optimistic Exploration in RL without Bonuses
> 
> Authors: Daniil Tiapkin and Denis Belomestny and Eric Moulines and Alexey Naumov and Sergey Samsonov and Yunhao Tang and Michal Valko and Pierre MENARD
> 
> Abstract: We propose the Bayes-UCBVI algorithm for reinforcement learning in tabular, stage-dependent, episodic Markov decision process: a natural extension of the Bayes-UCB algorithm by Kaufmann et al. 2012 for multi-armed bandits. Our method uses the quantile of a Q-value function posterior as upper confidence bound on the optimal Q-value function. For Bayes-UCBVI, we prove a regret bound of order $\tcO(\sqrt{H^3SAT})$ where $H$ is the length of one episode, $S$ is the number of states, $A$ the number of actions, $T$ the number of episodes, that matches the lower-bound of $\Omega(\sqrt{H^3SAT})$ up to poly-$\log$ terms in $H,S,A,T$ for a large enough $T$. To the best of our knowledge, this is the first algorithm that obtains an optimal dependence on the horizon $H$ (and $S$) \textit{without the need of an involved Bernstein-like bonus or noise.} Crucial to our analysis is a new fine-grained anti-concentration bound for a weighted Dirichlet sum that can be of independent interest. We then explain how Bayes-UCBVI can be easily extended beyond the tabular setting, exhibiting a strong link between our algorithm and Bayesian bootstrap (Rubin,1981).

> Oral: Tackling covariate shift with node-based Bayesian neural networks
> 
> Authors: Trung Trinh and Markus Heinonen and Luigi Acerbi and Samuel Kaski
> 
> Abstract: Bayesian neural networks (BNNs) promise improved generalisation under covariate shift by providing principled probabilistic representations of epistemic uncertainty. However, weight-based BNNs often struggle with high computational complexity of large scale architectures and datasets. Node-based BNNs have recently been introduced as scalable alternatives, which induce epistemic uncertainty by multiplying each hidden node with latent random variables, while learning a point-estimate of the weights. In this paper, we interpret these latent variables as implicit representations of simple and domain-agnostic input corruptions during training, producing BNNs performing well under covariate shift. We observe that the diversity of the implicit corruptions depends on the entropy of the latent variables, and propose a straightforward approach to increase the entropy of these variables during training. We evaluate the method on out-of-distribution image classification benchmarks, and show improved uncertainty estimation of node-based BNNs under covariate shift. As a side effect, the method also provides robustness against noisy training labels.


> Oral: A Simple yet Universal Strategy for Online Convex Optimization
> 
> Authors: Lijun Zhang and Guanghui Wang and Jinfeng Yi and Tianbao Yang
> 
> Abstract: Recently, several universal methods have been proposed for online convex optimization, and attain minimax rates for multiple types of convex  functions simultaneously. However, they need to design and optimize one surrogate loss for each type of functions, which makes it difficult to exploit the structure of the problem and utilize existing algorithms. In this paper, we propose a simple strategy for universal online convex optimization, which avoids these limitations. The key idea is to construct a set of experts to process the original online functions, and deploy a meta-algorithm over the linearized losses to aggregate predictions from experts. Specifically, the meta-algorithm is required to yield a second-order bound with excess losses, so that it can leverage strong convexity and exponential concavity to control the meta-regret. In this way, our strategy inherits the theoretical guarantee of any expert designed for strongly convex functions and exponentially concave functions, up to a double logarithmic factor. As a result, we can plug in off-the-shelf online solvers as black-box experts to deliver problem-dependent regret bounds. For general convex functions, it maintains the minimax optimality and also achieves a small-loss bound.


> Oral: Connect, Not Collapse: Explaining Contrastive Learning for Unsupervised Domain Adaptation
> 
> Authors: Kendrick Shen and Robbie Jones and Ananya Kumar and Sang Michael Xie and Jeff Z. HaoChen and Tengyu Ma and Percy Liang
> 
> Abstract: We consider unsupervised domain adaptation (UDA), where labeled data from a source domain (e.g., photographs) and unlabeled data from a target domain (e.g., sketches) are used to learn a classifier for the target domain. Conventional UDA methods (e.g., domain adversarial training) learn domain-invariant features to improve generalization to the target domain. In this paper, we show that contrastive pre-training, which learns features on unlabeled source and target data and then fine-tunes on labeled source data, is competitive with strong UDA methods. However, we find that contrastive pre-training does not learn domain-invariant features, diverging from conventional UDA intuitions. We theoretically analyze how contrastive pre-training can learn features that vary subtantially across domains but still generalize to the target domain. Our results suggest that domain invariance is not necessary for UDA. We empirically validate our theory on benchmark vision datasets.


> Oral: Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition
> 
> Authors: Haotao Wang and Aston Zhang and Yi Zhu and Shuai Zheng and Mu Li and Alex Smola and Zhangyang Wang
> 
> Abstract: Existing out-of-distribution (OOD) detection methods are typically benchmarked on training sets with balanced class distributions. However, in real-world applications, it is common for the training sets to have long-tailed distributions. In this work, we first demonstrate that existing OOD detection methods commonly suffer from significant performance drop when the train set is long-tail distributed. Through analysis, we posit that this is because the resultant models struggle to distinguish the minority tail-class in-distribution (ID) samples from OOD samples, making them more prone to be falsely detected as OOD. To solve this problem, we propose Partial and Asymmetric Supervised Contrastive Learning, which explicitly encourages the model to distinguish tail-class ID samples from OOD samples. To further boost ID classification accuracy, we propose Auxiliary Branch Finetuning, which uses two separate branches of BN and classification layers for anomaly detection and ID classification. The intuition is that ID and OOD anomaly data have different underlying distributions. Our method outperforms previous state-of-the-art method by $1.29\%$, $1.45\%$, $0.69\%$ anomaly detection false positive rate (FPR) and $3.24\%$, $4.06\%$, $7.89\%$ ID classification accuracy on CIFAR10-LT, CIFAR100-LT, and ImageNet-LT, respectively. Source code and pre-trained models will be released.

> Oral: Planning with Diffusion for Flexible Behavior Synthesis
> 
> Authors: Michael Janner and Yilun Du and Josh Tenenbaum and Sergey Levine
> 
> Abstract: Model-based reinforcement learning methods often use learning only for the purpose of recovering an approximate dynamics model, offloading the rest of the decision-making work to classical trajectory optimizers.While conceptually simple, this combination has a number of empirical shortcomings, suggesting that learned models may not be well-suited to standard trajectory optimization.In this paper, we consider what it would look like to fold as much of the trajectory optimization pipeline as possible into the modeling problem, such that sampling from the model and planning with it become nearly identical.The core of our technical approach lies in a diffusion probabilistic model that plans by iteratively denoising trajectories.We show how classifier-guided sampling and image inpainting can be reinterpreted as coherent planning strategies, explore the unusual and useful properties of diffusion-based planning methods, and demonstrate the effectiveness of our framework in control settings that emphasize long-horizon decision-making and test-time flexibility.


> Oral: Out-of-Distribution Detection with Posterior Sampling
> 
> Authors: Yifei Ming and Ying Fan and Sharon Li
> 
> Abstract: Out-of-distribution (OOD) detection is indispensable for machine learning models deployed in the open world. Recently, the use of an auxiliary outlier dataset during training (also known as outlier exposure) has shown promising performance. As the sample space for potential OOD data can be prohibitively large, sampling informative outliers is essential. In this work, we propose a novel posterior sampling based outlier mining framework, POEM, which facilitates efficient use of outlier data and promotes learning a compact decision boundary between ID and OOD data for improved detection. We show that POEM establishes state-of-the-art performance on common benchmarks. Compared to the current best method that uses a greedy sampling strategy, POEM improves the relative performance by 42.0% and 24.2% (FPR95) on CIFAR-10 and CIFAR-100, respectively. We further provide theoretical insights on the effectiveness of POEM for OOD detection.


> Oral: A Minimax Learning Approach to Off-Policy Evaluation in Partially Observable Markov Decision Processes
> 
> Authors: Chengchun Shi and Masatoshi Uehara and Jiawei Huang and Nan Jiang
> 
> Abstract: We consider off-policy evaluation (OPE) in Partially Observable Markov Decision Processes (POMDPs), where the evaluation policy depends only on observable variables and the behavior policy depends on unobservable latent variables. Existing works either assume no unmeasured confounders, or focus on settings where both the observation and the state spaces are tabular. In this work, we first propose novel identification methods for OPE in POMDPs with latent confounders, by introducing bridge functions that link the target policy's value and the observed data distribution. We next propose minimax estimation methods for learning these bridge functions, and construct three estimators based on these estimated bridge functions, corresponding to a value function-based estimator, a marginalized importance sampling estimator, and a doubly-robust estimator. Our proposal permits general function approximation and is thus applicable to settings with continuous or large observation/state spaces. The nonasymptotic and asymptotic properties of the proposed estimators are investigated in detail.


> Oral: Federated Reinforcement Learning: Communication-Efficient Algorithms and Convergence Analysis
> 
> Authors: sajad khodadadian and PRANAY SHARMA and Gauri Joshi and Siva Maguluri
> 
> Abstract: Since reinforcement learning algorithms are notoriously data-intensive, the task of sampling observations from the environment is usually split across multiple agents. However, transferring these observations from the agents to a central location can be prohibitively expensive in terms of the communication cost, and it can also compromise the privacy of each agent's local behavior policy. In this paper, we consider a federated reinforcement learning framework where multiple agents collaboratively learn a global model, without sharing their individual data and policies. Each agent maintains a local copy of the model and updates it using locally sampled data. Although having N agents enables the sampling of N times more data, it is not clear if it leads to proportional convergence speed-up. We propose federated versions of on-policy TD, off-policy TD and Q-learning, and analyze their convergence. For all these algorithms, to the best of our knowledge, we are the first to consider Markovian noise and multiple local updates, and prove a linear convergence speedup with respect to the number of agents. To obtain these results, we show that federated TD and Q-learning are special cases of a general framework for federated stochastic approximation with Markovian noise, and we leverage this framework to provide a unified convergence analysis that applies to all the algorithms.


> Oral: Batched Dueling Bandits
> 
> Authors: Arpit Agarwal and Rohan Ghuge and viswanath nagarajan
> 
> Abstract: The K-armed dueling bandit problem, where the feedback is in the form of noisy pairwise comparisons, has been widely studied. Previous works have only focused on the sequential setting where the policy adapts after every comparison. However, in many applications such as search ranking and recommendation systems, it is preferable to perform comparisons in a limited number of parallel batches. We study the  batched K-armed dueling bandit problem under two standard settings: (i) existence of a Condorcet winner, and (ii) strong stochastic transitivity and  stochastic  triangle inequality. For both settings, we obtain algorithms with a smooth trade-off between the number of batches and regret. Our regret bounds  match the best known sequential regret bounds (up to poly-logarithmic factors),  using only a logarithmic number of batches. We complement our regret analysis with a nearly-matching lower bound. Finally, we also validate our theoretical results via experiments on synthetic and real data.


> Oral: Stable Conformal Prediction Sets
> 
> Authors: Eugene Ndiaye
> 
> Abstract: When one observes a sequence of variables $(x_1, y_1), \ldots, (x_n, y_n)$, Conformal Prediction (CP) is a methodology that allows to estimate a confidence set for $y_{n+1}$ given $x_{n+1}$ by merely assuming that the distribution of the data is exchangeable. CP sets have guaranteed coverage for any finite population size $n$. While appealing, the computation of such set turns out to be infeasible in general, \eg when the unknown variable $y_{n+1}$ is continuous. The bottleneck is that it is based on a procedure that readjusts a prediction model on data where we replaced the unknown target by all its possible values in order to select the most probable one. This requires computing an infinite number of models, which often makes it intractable. We combine CP techniques with algorithmic stability bounds to derive a prediction set computable with a single model fit. We demonstrate that our proposed confidence set does not lose any coverage guarantees while avoiding the need for data splitting as currently done in the literature. We perform some numerical experiments that illustrate the tightness of our estimation when the sample size is sufficiently large.

> Oral: Unified Scaling Laws for Routed Language Models
> 
> Authors: Aidan Clark and Diego de Las Casas and Aurelia Guy and Arthur Mensch and Michela Paganini and Jordan Hoffmann and Bogdan Damoc and Blake Hechtman and Trevor Cai and Sebastian Borgeaud and George van den Driessche and Eliza Rutherford and Tom Hennigan and Matthew Johnson and Albin Cassirer and Chris Jones and Elena Buchatskaya and David Budden and Laurent Sifre and Simon Osindero and Oriol Vinyals and Marc'Aurelio Ranzato and Jack Rae and Erich Elsen and Koray Kavukcuoglu and Karen Simonyan
> 
> Abstract: The performance of a language model has been shown to be effectively modeled as a power-law in its parameter count. Here we study the scaling behaviors of Routing Networks: architectures that conditionally use only a subset of their parameters while processing an input. For these models, parameter count and computational requirement form two independent axes along which an increase leads to better performance. In this work we derive and justify scaling laws defined on these two variables which generalize those known for standard language models and describe the performance of a wide range of routing architectures trained via three different techniques. Afterwards we provide two applications of these laws: first deriving an Effective Parameter Count along which all models scale at the same rate, and then using the scaling coefficients to give a quantitative comparison of the three routing techniques considered. Our analysis derives from an extensive evaluation of Routing Networks across five orders of magnitude of size, including models with hundreds of experts and hundreds of billions of parameters.


> Oral: Path-Gradient Estimators for Continuous Normalizing Flows
> 
> Authors: Lorenz Vaitl and Kim Nicoli and Shinichi Nakajima and Pan Kessel
> 
> Abstract: Recent work has established a path-gradient estimator for simple variational Gaussian distributions and has argued that the path-gradient is particularly beneficial in the regime in which the variational distribution approaches the exact target distribution. In many applications, this regime can however not be reached by a simple Gaussian variational distribution. In this work, we overcome this crucial limitation by proposing a path-gradient estimator for the considerably more expressive variational family of continuous normalizing flows. We outline an efficient algorithm to calculate this estimator and establish its superior performance empirically. 


> Oral: Rethinking Image-Scaling Attacks: The Interplay Between Vulnerabilities in Machine Learning Systems
> 
> Authors: Yue Gao and Ilia Shumailov and Kassem Fawaz
> 
> Abstract: As real-world images come in varying sizes, the machine learning model is part of a larger system that includes an upstream image scaling algorithm. In this paper, we investigate the interplay between vulnerabilities of the image scaling procedure and machine learning models in the decision-based black-box setting. We propose a novel sampling strategy to make a black-box attack exploit vulnerabilities in scaling algorithms, scaling defenses, and the final machine learning model in an end-to-end manner. Based on this scaling-aware attack, we reveal that most existing scaling defenses are ineffective under threat from downstream models. Moreover, we empirically observe that standard black-box attacks can significantly improve their performance by exploiting the vulnerable scaling procedure. We further demonstrate this problem on a commercial Image Analysis API with transfer-based black-box attacks.


> Oral: Causal Dynamics Learning for Task-Independent State Abstraction
> 
> Authors: Zizhao Wang and Xuesu Xiao and Zifan Xu and Yuke Zhu and Peter Stone
> 
> Abstract: Learning dynamics models accurately is an important goal for Model-Based Reinforcement Learning (MBRL), but most MBRL methods learn a dense dynamics model which is vulnerable to spurious correlations and therefore generalizes poorly to unseen states. In this paper, we introduce Causal Dynamics Learning for Task-Independent State Abstraction (CDL), which first learns a theoretically proved causal dynamics model that removes unnecessary dependencies between state variables and the action, thus generalizing well to unseen states. A state abstraction can then be derived from the learned dynamics, which not only improves sample efficiency but also applies to a wider range of tasks than existing state abstraction methods. Evaluated on two simulated environments and downstream tasks, both the dynamics model and policies learned by the proposed method generalize well to unseen states and the derived state abstraction improves sample efficiency compared to learning without it.


> Oral: Generating 3D Molecules for Target Protein Binding
> 
> Authors: Meng Liu and Youzhi Luo and Kanji Uchino and Koji Maruhashi and Shuiwang Ji
> 
> Abstract: A fundamental problem in drug discovery is to design molecules that bind to specific proteins. To tackle this problem using machine learning methods, here we propose a novel and effective framework, known as GraphBP, to generate 3D molecules that bind to given proteins by placing atoms of specific types and locations to the given binding site one by one. In particular, at each step, we first employ a 3D graph neural network to obtain geometry-aware and chemically informative representations from the intermediate contextual information. Such context includes the given binding site and atoms placed in the previous steps. Second, to preserve the desirable equivariance property, we select a local reference atom according to the designed auxiliary classifiers and then construct a local spherical coordinate system. Finally, to place a new atom, we generate its atom type and relative location \emph{w.r.t.} the constructed local coordinate system via a flow model. We also consider generating the variables of interest sequentially to capture the underlying dependencies among them. Experiments demonstrate that our GraphBP is effective to generate 3D molecules with binding ability to target protein binding sites.


> Oral: Generalized Strategic Classification and the Case of Aligned Incentives
> 
> Authors: Sagi Levanon and Nir Rosenfeld
> 
> Abstract: Strategic classification studies learning in settings where self-interested users can strategically modify their features to obtain favorable predictive outcomes. A key working assumption, however, is that favorable'' always meanspositive''; this may be appropriate in some applications (e.g., loan approval), but amounts to a fairly narrow view what user interests can be. In this work we argue for a broader perspective on what can account for strategic user behavior, and propose and study a flexible model of generalized strategic classification. Our generalized model subsumes most current models, but includes other novel settings; among these, we identify and target one intriguing sub-class of problems in which the interests of users and the system are aligned. For this cooperative setting, we provide an in-depth analysis, and propose a practical learning approach that is effective and efficient. Returning to our fully generalized model, we show how our results and approach can extend to the most general case. We conclude with a set of experiments that empirically demonstrate the utility of our approach.


> Oral: Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations
> 
> Authors: Michael Zhang and Nimit Sohoni and Hongyang Zhang and Chelsea Finn and Christopher Re
> 
> Abstract: Spurious correlations pose a major challenge for robust machine learning. Models trained with empirical risk minimization (ERM) may learn to rely on correlations between class labels and spurious attributes, leading to poor performance on data groups without these correlations. This is challenging to address when the spurious attribute labels are unavailable. To improve worst-group performance on spuriously correlated data without training attribute labels, we propose Correct-N-Contrast (CNC), a contrastive approach to directly learn representations robust to spurious correlations. As ERM models can be good spurious attribute predictors, CNC works by (1) using a trained ERM model’s outputs to identify samples with the same class but dissimilar spurious features, and (2) training a robust model with contrastive learning to learn similar representations for these samples. To support CNC, we introduce new connections between worst-group error and a representation alignment loss that CNC aims to minimize. We empirically observe that worst-group error closely tracks with alignment loss, and prove that the alignment loss over a class helps upper-bound the class's worst-group vs. average error gap. On popular benchmarks, CNC reduces alignment loss drastically, and achieves state-of-the-art worst-group accuracy by 3.6% average absolute lift. CNC is also competitive with oracle methods that require group labels.


> Oral: Causal Conceptions of Fairness and their Consequences
> 
> Authors: Hamed Nilforoshan and Johann Gaebler and Ravi Shroff and Sharad Goel
> 
> Abstract: Recent work highlights the role of causality in designing equitable decision-making algorithms. It is not immediately clear, however, how existing causal conceptions of fairness relate to one another, nor what the consequences are of using these definitions as design principles.Here, we first assemble and categorize popular causal definitions of algorithmic fairness into two broad families: (1) those that constrain the effects of decisions on counterfactual disparities; and (2) those that constrain the effects of protected characteristics---like race and gender---on decisions. We then show, analytically and empirically, that both families of definitions typically result in (strongly) Pareto dominated decision policies, meaning there is an alternative, unconstrained policy favored by every stakeholder with preferences drawn from a large, natural class. For example, in the case of college admissions decisions, policies constrained to satisfy causal fairness definitions would be disfavored by every stakeholder with neutral or positive preferences for both academic preparedness and diversity.Indeed, under a prominent definition of causal fairness, we prove the resulting policies require admitting all students with the same probability, regardless of academic qualifications or group membership.


> Oral: Refined Convergence Rates for Maximum Likelihood Estimation under Finite Mixture Models
> 
> Authors: Tudor Manole and Nhat Ho
> 
> Abstract: We revisit convergence rates for maximum likelihood estimation (MLE) under finite mixture models. The Wasserstein distance has become a standard loss function for the analysis of parameter estimation in these models, due in part to its ability to circumvent label switching and to accurately characterize the behaviour of fitted mixture components with vanishing weights. However, the Wasserstein metric is only able to capture the worst-case convergence rate among the remaining fitted mixture components. We demonstrate that when the log-likelihood function is penalized to discourage vanishing mixing weights, stronger loss functions can be derived to resolve this shortcoming of the Wasserstein distance. These new loss functions accurately capture the heterogeneity in convergence rates of fitted mixture components, and we use them to sharpen existing pointwise and uniform convergence rates in various classes of mixture models. In particular,  these results imply that a subset of the components of the penalized MLE typically converge significantly faster than could have been anticipated from past work. We further show that some of these conclusions extend to the traditional MLE. Our theoretical findings are supported by a simulation study to illustrate these improved convergence rates. 


> Oral: On the Convergence of Inexact Predictor-Corrector Methods for Linear Programming
> 
> Authors: Gregory Dexter and Agniva Chowdhury and Haim Avron and Petros Drineas
> 
> Abstract: Interior point methods (IPMs) are a common approach for solving linear programs (LPs) with strong theoretical guarantees and solid empirical performance. The time complexity of these methods is dominated by the cost of solving a linear system of equations at each iteration. In common applications of linear programming, particularly in machine learning and scientific computing, the size of this linear system can become prohibitively large, requiring the use of iterative solvers, which provide an approximate solution to the linear system. However, approximately solving the linear system at each iteration of an IPM invalidates the theoretical guarantees of common IPM analyses. To remedy this, we theoretically and empirically analyze (slightly modified) predictor-corrector IPMs when using approximate linear solvers: our approach guarantees that, when certain conditions are satisfied, the number of IPM iterations does not increase and that the final solution remains feasible. We also provide practical instantiations of approximate linear solvers that satisfy these conditions for special classes of constraint matrices using randomized linear algebra.


> Oral: Large Batch Experience Replay
> 
> Authors: Thibault Lahire and Matthieu Geist and Emmanuel Rachelson
> 
> Abstract: Several algorithms have been proposed to sample non-uniformly the replay buffer of deep Reinforcement Learning (RL) agents to speed-up learning, but very few theoretical foundations of these sampling schemes have been provided. Among others, Prioritized Experience Replay appears as a hyperparameter sensitive heuristic, even though it can provide good performance. In this work, we cast the replay buffer sampling problem as an importance sampling one for estimating the gradient. This allows deriving the theoretically optimal sampling distribution, yielding the best theoretical convergence speed.Elaborating on the knowledge of the ideal sampling scheme, we exhibit new theoretical foundations of Prioritized Experience Replay. The optimal sampling distribution being intractable, we make several approximations providing good results in practice and introduce, among others, LaBER (Large Batch Experience Replay), an easy-to-code and efficient method for sampling the replay buffer. LaBER, which can be combined with Deep Q-Networks, distributional RL agents or actor-critic methods, yields improved performance over a diverse range of Atari games and PyBullet environments, compared to the base agent it is implemented on and to other prioritization schemes.


> Oral: REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer
> 
> Authors: Xingyu Liu and Deepak Pathak and Kris Kitani
> 
> Abstract: Popular paradigm in robotic learning is to train a policy from scratch for every new robot. This is not only inefficient but often impractical for complex robots. In this work, we consider the problem of transfer policy across two different robots with significantly different parameters such as kinematics and morphology. Existing approaches that train a new policy by matching the action or state transition distribution, including imitation learning methods, fail due to optimal action and/or state distribution being different in different robots. In this paper, we propose a novel method of using continuous evolutionary models for robotic policy transfer. We interpolate between the source robot and the target robot by finding a continuous evolutionary change of robot parameters. An expert policy on the source robot is transferred through iteratively finetuning on the intermediate robots that gradually evolve to the target robot. Experiments show that the proposed continuous evolutionary model can effectively transfer the policy across robots and achieve superior sample efficiency on new robots. The proposed method is especially advantageous in sparse reward settings where exploration can be significantly reduced.


> Oral: Efficient Contextual Bandits with CVaR Regret
> 
> Authors: Yinglun Zhu and Paul Mineiro
> 
> Abstract: In the (contextual) bandit literature, alternative notions of regret have been proposed to deal with large or even continuous action space problems where obtaining the standard regret guarantee is hopeless without additional assumptions. Specifically, the bandit literature proposes quantile regret, while the contextual bandit literature proposes smoothed regret. These regret definitions avoid the need for assumptions such as linearity or smoothness.  We propose to compete with the Conditional Value at Risk at quantile $h$ (CVaR regret), which dominates both quantile regret and smoothed regret. We present a computationally and statistically efficient algorithm for CVaR regret at any fixed quantile which works with general function approximation under standard supervised oracles. We also present an adaptive algorithm for simultaneously competing with all quantiles. Our algorithms recover the previous minimax/Pareto optimal guarantees under the \emph{standard} regret definition, e.g., in bandit problems with multiple best arms and Lipschitz/H{\"o}lder bandits. We conduct large-scale empirical evaluations to demonstrate the efficacy of our proposed algorithms.

> Oral: The Importance of Non-Markovianity in Maximum State Entropy Exploration
> 
> Authors: Mirco Mutti and Riccardo De Santi and Marcello Restelli
> 
> Abstract: In the maximum state entropy exploration framework, an agent interacts with a reward-free environment to learn a policy that maximizes the entropy of the expected state visitations it is inducing. Hazan et al. (2019) noted that the class of Markovian stochastic policies is sufficient for the maximum state entropy objective, and exploiting non-Markovianity is generally considered pointless in this setting. In this paper, we argue that non-Markovianity is instead paramount for maximum state entropy exploration in a finite-sample regime. Especially, we recast the objective to target the expected entropy of the induced state visitations in a single trial. Then, we show that the class of non-Markovian deterministic policies is sufficient for the introduced objective, while Markovian policies suffer non-zero regret in general. However, we prove that the problem of finding an optimal non-Markovian policy is NP-complete. Despite this negative result, we discuss avenues to address the problem in a tractable way and how non-Markovian exploration could benefit the sample efficiency of online reinforcement learning in future works.


> Oral: Robust Training of Neural Networks using Scale Invariant Architectures
> 
> Authors: Zhiyuan Li and Srinadh Bhojanapalli and Manzil Zaheer and Sashank Jakkam Reddi and Sanjiv Kumar
> 
> Abstract: In contrast to SGD, adaptive gradient methods like Adam allow robust training of modern deep networks, especially large language models. However, the use of adaptivity not only comes at the cost of extra memory but also raises the fundamental question: can non-adaptive methods like SGD enjoy similar benefits?In this paper, we provide an affirmative answer to this question by proposing to achieve both robust and memory-efficient training via the following general recipe: (1) modify the architecture and make it scale invariant, (2) train with SGD and weight decay, and optionally (3) clip the global gradient norm proportional to weight norm multiplied by $\sqrt{\frac{2\lambda}{\eta}}$, where $\eta$ is learning rate and $\lambda$ is weight decay. We show that this general approach is robust to rescaling of parameter and loss by proving that its convergence only depends logarithmically on the scale of initialization and loss, whereas the standard SGD might not even converge for many initializations. Following our recipe, we design a scale invariant version of BERT, called SIBERT, which when trained simply by vanilla SGD achieves performance comparable to BERT trained by adaptive methods like Adam on downstream tasks.

> Oral: Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models.
> 
> Authors: Abhineet Agarwal and Yan Shuo Tan and omer ronen and Chandan Singh and Bin Yu
> 
> Abstract: Decision trees and random forests (RF) are a cornerstone of modern machine learning practice. Due to their tendency to overfit, trees are typically regularized by a variety of techniques that modify their structure (e.g. pruning). We introduce Hierarchical Shrinkage (HS), a post-hoc algorithm which regularizes the tree not by altering its structure, but by shrinking the prediction over each leaf toward the sample means over each of its ancestors, with weights depending on a single regularization parameter and the number of samples in each ancestor.  Since HS is a post-hoc method, it is extremely fast, compatible with any tree-growing algorithm and can be used synergistically with other regularization techniques. Extensive experiments over a wide variety of real-world datasets show that HS substantially increases the predictive performance of decision trees even when used in conjunction with other regularization techniques. Moreover, we find that applying HS to individual trees in a RF often improves its accuracy and interpretability by simplifying and stabilizing decision boundaries and SHAP values. We further explain HS by showing that it to be equivalent to ridge regression on a basis that is constructed of decision stumps associated to the internal nodes of a tree. All code and models are released in a full-fledged package available on Github


> Oral: G-Mixup: Graph Data Augmentation for Graph Classification
> 
> Authors: Xiaotian Han and Zhimeng Jiang and Ninghao Liu and Xia Hu
> 
> Abstract: This work develops \emph{mixup for graph data}. Mixup has shown superiority in improving the generalization and robustness of neural networks by interpolating features and labels between two random samples. Traditionally, Mixup can work on regular, grid-like, and Euclidean data such as image or tabular data. However, it is challenging to directly adopt Mixup to augment graph data because different graphs typically: 1) have different numbers of nodes; 2) are not readily aligned; and 3) have unique typologies in non-Euclidean space. To this end, we propose $\mathcal{G}$-Mixup to augment graphs for graph classification by interpolating the generator (i.e., graphon) of different classes of graphs. Specifically, we first use graphs within the same class to estimate a graphon. Then, instead of directly manipulating graphs, we interpolate graphons of different classes in the Euclidean space to get mixed graphons, where the synthetic graphs are generated through sampling based on the mixed graphons. Extensive experiments show that $\mathcal{G}$-Mixup substantially improves the generalization and robustness of GNNs.

> Oral: Bayesian Continuous-Time Tucker Decomposition
> 
> Authors: Shikai Fang and Akil Narayan and Robert Kirby and Shandian Zhe
> 
> Abstract: Tensor decomposition is a dominant framework for multiway data analysis and prediction. Although practical data often contains timestamps for the observed entries, existing tensor decomposition approaches  overlook or  under-use this valuable time information. They either drop the timestamps or bin them into crude steps and hence ignore the temporal dynamics within each step or use simple parametric time coefficients. To overcome these limitations, we propose Bayesian Continuous-Time Tucker Decomposition. We model the tensor-core of the classical Tucker decomposition as a time-varying function, and place a Gaussian process prior to flexibly estimate all kinds of temporal dynamics. In this way, our model maintains the interpretability while is flexible enough to capture various complex temporal relationships between the tensor nodes.  For efficient and high-quality posterior inference, we use the stochastic differential equation (SDE) representation of temporal GPs to build an equivalent state-space prior, which avoids huge kernel matrix computation and sparse/low-rank approximations. We then use Kalman filtering, RTS smoothing, and conditional moment matching to develop a scalable message passing inference algorithm. We show the advantage of our method in simulation and several real-world applications. 


> Oral: Improved Rates for Differentially Private Stochastic Convex Optimization with Heavy-Tailed Data
> 
> Authors: Gautam Kamath and Xingtu Liu and Huanyu Zhang
> 
> Abstract: We study stochastic convex optimization with heavy-tailed data under the constraint of differential privacy (DP). Most prior work on this problem is restricted to the case where the loss function is Lipschitz. Instead, as introduced by Wang, Xiao, Devadas, and Xu~\cite{WangXDX20}, we study general convex loss functions with the assumption that the distribution of gradients has bounded $k$-th moments. We provide improved upper bounds on the excess population risk under concentrated DP for convex and strongly convex loss functions. Along the way, we derive new algorithms for private mean estimation of heavy-tailed distributions, under both pure and concentrated DP. Finally, we prove nearly-matching lower bounds for private stochastic convex optimization with strongly convex losses and mean estimation, showing new separations between pure and concentrated DP.

> Oral: Active fairness auditing
> 
> Authors: Tom Yan and Chicheng Zhang
> 
> Abstract: The fast spreading adoption of machine learning by companies across industries poses significant regulatory challenges. One such challenge is scalability: how can regulatory bodies efficiently \emph{audit} these ML models, ensuring that they are fair? In this paper, we initiate the study of query-based auditing algorithms that can estimate the demographic parity of ML models in a query-efficient manner. We propose an optimal deterministic algorithm, as well as a practical randomized, oracle-efficient algorithm with comparable guarantees. Furthermore, we make inroads into understanding the optimal query complexity of randomized active fairness estimation algorithms. Our first exploration of active fairness estimation aims to put AI governance on firmer theoretical foundations.


> Oral: Tight and Robust Private Mean Estimation with Few Users
> 
> Authors: Shyam Narayanan and Vahab Mirrokni and Hossein Esfandiari
> 
> Abstract: In this work, we study high-dimensional mean estimation under user-level differential privacy, and design an $(\epsilon,\delta)$-differentially private mechanism using as few users as possible. In particular, we provide a nearly optimal trade-off between the number of users and the number of samples per user required for private mean estimation, even when the number of users is as low as $O(\frac{1}{\epsilon}\log\frac{1}{\delta})$. Interestingly, this bound is independent of the dimension, unlike the previous work that depends polynomially on the dimension. This resolves a problem first proposed by Amin et al.~\yrcite{amin2019biasvarianceuser}. Moreover, our mechanism is robust against corruptions in up to $49\%$ of the users. Finally, our results also apply to optimal algorithms for privately learning discrete distributions with few users, answering a question of Liu et al.~\yrcite{liu2020discrete}, and a broader range of problems such as stochastic convex optimization and a variant of stochastic gradient descent via a reduction to differentially private mean estimation.

> Oral: Do Differentiable Simulators Give Better Gradients for Policy Optimization?
> 
> Authors: Hyung Ju Suh and Max Simchowitz and Kaiqing Zhang and Russ Tedrake
> 
> Abstract: Differentiable simulators promise faster computation time for reinforcement learning by replacing zeroth-order gradient estimates of a stochastic objective with an estimate based on first-order gradients. However, it is yet unclear what factors decide the performance of the two estimators on complex landscapes that involve long-horizon planning and control on physical systems, despite the crucial relevance of this question for the utility of differentiable simulators. We show that characteristics of certain physical systems, such as stiffness or discontinuities, may compromise the efficacy of the first-order estimator, and analyze this phenomenon through the lens of bias and variance. We additionally propose an $\alpha$-order gradient estimator, with $\alpha \in [0,1]$, which correctly utilizes exact gradients to combine the efficiency of first-order estimates with the robustness of zero-order methods. We demonstrate the pitfalls of traditional estimators and the advantages of the $\alpha$-order estimator on some numerical examples.

> Oral: Learning Bellman Complete Representations for Offline Policy Evaluation
> 
> Authors: Jonathan Chang and Kaiwen Wang and Nathan Kallus and Wen Sun
> 
> Abstract: We study representation learning for Offline Reinforcement Learning (RL), focusing on the important sub-task of Offline Policy Evaluation (OPE). Recent work shows that, in contrast to supervised learning, realizability of the Q-function is not enough for learning it. Two sufficient conditions for sample-efficient OPE are Bellman completeness and coverage. Achieving Bellman completeness is nontrivial since, unlike realizability, it is not monotonic: it may break by making representations richer. Prior work often assumes that representations satisfying these conditions are given,with results being mostly theoretical in nature. In this work, we propose a novel algorithm that directly learns a representation that is both approximately Bellman complete and provides good coverage. Once learned, we perform OPE using theLeast Square Policy Evaluation (LSPE) algorithm using linear functions in our learned representation. We present an end-to-end theoretical analysis, showing that our two-stage OPE procedure enjoys polynomial sample complexity provided some representation in the rich class considered is Bellman complete. Empirically, we compare to other representation learning techniques that were previously developed for off-policy RL approaches (e.g., CURL, SPR) on a set of image-based continuous control tasks. Our experimental results demonstrate that our learned representation enables better OPE in such difficult tasks.


> Oral: Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning
> 
> Authors: Utku Evci and Vincent Dumoulin and Hugo Larochelle and Michael Mozer
> 
> Abstract: Transfer-learning methods aim to improve performance in a data-scarce target domain using a model pretrained on a data-rich source domain. A cost-efficient strategy, linear probing, involves freezing the source model and training a new classification head for the target domain. This strategy is outperformed by a more costly but state-of-the-art method -- fine-tuning all parameters of the source model to the target domain -- possibly because fine-tuning allows the model to leverage useful information from intermediate layers which is otherwise discarded by the later previously trained layers. We explore the hypothesis that these intermediate layers might be directly exploited. We propose a method, Head-to-Toe probing (Head2Toe), that selects features from all layers of the source model to train a classification head for the target-domain. In evaluations on the Visual Task Adaptation Benchmark-1k, Head2Toe matches performance obtained with fine-tuning on average while reducing training and storage cost hundred folds or more, but critically, for out-of-distribution transfer, Head2Toe outperforms fine-tuning. Code used in our experiments can be found in supplementary materials.


> Oral: Equivariant Diffusion for Molecule Generation in 3D
> 
> Authors: Emiel Hoogeboom and Víctor Garcia Satorras and Clément Vignac and Max Welling
> 
> Abstract: This work introduces a diffusion model for molecule generation in 3D that is equivariant to Euclidean transformations. Our E(3) Equivariant Diffusion Model (EDM) learns to denoise a diffusion process with an equivariant network that jointly operates on both continuous (atom coordinates) and categorical features (atom types). In addition, we provide a probabilistic analysis which admits likelihood computation of molecules using our model. Experimentally, the proposed method significantly outperforms previous 3D molecular generative methods regarding the quality of generated samples and the efficiency at training time. 


> Oral: ModLaNets: Learning Generalisable Dynamics via Modularity and Physical Inductive Bias
> 
> Authors: Yupu Lu and Shijie Lin and Guanqi Chen and Jia Pan
> 
> Abstract: Deep learning models are able to approximate one specific dynamical system, but struggle at learning generalisable dynamics, where dynamical systems obey the same laws of physics but contain different numbers of elements (e.g., double- and triple-pendulum systems). To relieve this issue, we proposed the Modular Lagrangian Network (ModLaNet), a structural neural network framework with modularity and physical inductive bias. This framework focuses on modelling the energy of each element using modularity and then constructing the target dynamical systems via Lagrangian mechanics. Modularity is beneficial at reusing the trained networks, reducing the scale of networks and datasets. As a result, our framework can learn from the dynamics of simpler systems and extend to more complex ones, which is not feasible using other relevant physics-informed neural networks. We tested our framework on modelling double-pendulum or three-body systems and achieved the best performance compared with counterparts. We also reorganised our frameworks as extensions to model multi-pendulum and multi-body systems, demonstrating the intriguing reusable feature of our framework.


> Oral: Sublinear-Time Clustering Oracle for Signed Graphs
> 
> Authors: Stefan Neumann and Pan Peng
> 
> Abstract: Social networks are often modeled using signed graphs, where vertices correspond to users and edges have a sign that indicates whether an interaction between users was positive or negative. The arising signed graphs typically contain a clear community structure in the sense that the graph can be partitioned into a small number of polarized communities, each defining a sparse cut and indivisible into smaller polarized sub-communities. We provide a local clustering oracle for signed graphs with such a clear community structure, that can answer membership queries, i.e., “Given a vertex v, which community does v belong to?”, in sublinear time by reading only a small portion of the graph. Formally, when the graph has bounded maximum degree and the number of communities is at most O(log n), then with O(sqrt(n)poly(1/𝜖, log n)) preprocessing time, our oracle can answer each membership query in O(sqrt(n)poly(1/𝜖, log n)) time, and it correctly classifies a (1 − 𝜖)-fraction of vertices w.r.t. a set of hidden planted ground-truth communities. Our oracle is desirable in applications where the clustering information is needed for only a small number of vertices. Previously, such local clustering oracles were only known for unsigned graphs; our generalization to signed graphs requires a number of new ideas and gives a novel spectral analysis of the behavior of random walks with signs. We evaluate our algorithm for constructing such an oracle and answering membership queries on both synthetic and real-world datasets, validating its performance in practice.


> Oral: 3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design
> 
> Authors: Yinan Huang and Xingang Peng and Jianzhu Ma and Muhan Zhang
> 
> Abstract: Deep learning has achieved tremendous success in designing novel chemical compounds with desirable pharmaceutical properties. In this work, we focus on a new type of drug design problem --- generating a small 'linker' to physically attach two independent molecules with their distinct functions, which has not received much attention in this community. The main computational challenges include: 1) the generation of linkers is conditional on the two given molecules, in contrast to generating full molecules from scratch in previous works; 2) linkers heavily depend on the anchor atoms of the two molecules to be connected, which are not known beforehand; 3) 3D structures and orientations of the molecules need to be considered in the designing process to avoid atom clashes, for which invariance and equivariance to E(3) group are necessary. To address these problems, we propose a conditional generative model, named 3DLinker, which is able to predict anchor atoms and jointly generate linker molecular graphs and their 3D structures to connect the molecules based on an E(3) equivariant graph variational autoencoder. So far as we know, there are no previous models that can achieve this task. We compare our model with multiple conditional generative models modified from other molecular design tasks and find that our model has a significantly higher rate in recovering molecular graphs, and more importantly, accurately predicting the 3D coordinate of each atom.


> Oral: Generalised Policy Improvement with Geometric Policy Composition
> 
> Authors: Shantanu Thakoor and Mark Rowland and Diana Borsa and Will Dabney and Remi Munos and Andre Barreto
> 
> Abstract: We introduce a method for policy improvement that interpolates between the greedy approach of value-based reinforcement learning (RL) and the full planning approach typical of model-based RL. The new method builds on the concept of a geometric horizon model (GHM, also known as a \gamma-model), which models the discounted state-visitation distribution of a given policy. We show that we can evaluate any non-Markov policy that switches between a set of base Markov policies with fixed probability by a careful composition of the base policy GHMs, without any additional learning. We can then apply generalised policy improvement (GPI) to collections of such non-Markov policies to obtain a new Markov policy that will in general outperform its precursors. We provide a thorough theoretical analysis of this approach, develop applications to transfer and standard RL, and empirically demonstrate its effectiveness over standard GPI on a challenging deep RL continuous control task. We also provide an analysis of GHM training methods, proving a novel convergence result regarding previously proposed methods and showing how to train these models stably in deep RL settings.


> Oral: Measuring Representational Robustness of Neural Networks Through Shared Invariances
> 
> Authors: Vedant Nanda and Till Speicher and Camila Kolling and John P Dickerson and Krishna Gummadi and Adrian Weller
> 
> Abstract: Robustness has emerged as a key consideration in the study of machine learning models, asserting that a model's output is invariant to certain perturbations of its input. One goal is to study the relative robustness of two models, i.e. to assert that one model will not make mistakes on examples that another model (or a human) gets right. Currently, only a few methods are suitable to compare models directly, the most prominent of which are representation similarity metrics such as CKA and SVCCA. However, we demonstrate empirically that these representation similarity metrics cannot be used reliably to make robustness judgements. Based on this insight, we develop a new directional metric to compare the relative robustness of models, that measures how well a target model preserves the invariances of a reference model. We show that our measure retains the desirable properties of previous similarity metrics, but also allows us to make statements about the shared invariance of models. With the help of our measure, we are able to gain insights about how shared invariances vary with changes in weight initialization, architecture, loss, and training dataset.


> Oral: Scalable MCMC Sampling for Nonsymmetric Determinantal Point Processes
> 
> Authors: Insu Han and Mike Gartrell and Elvis Dohmatob and Amin Karbasi
> 
> Abstract: A determinantal point process (DPP) is an elegant model that assigns a probability to every subset of a collection of $n$ items.  While conventionally a DPP is parameterized by a symmetric kernel matrix, removing this symmetry constraint, resulting in nonsymmetric DPPs (NDPPs), leads to significant improvements in modeling power and predictive performance.  Recent work has studied an approximate Markov chain Monte Carlo (MCMC) sampling algorithm for NDPPs restricted to size-$k$ subsets (called $k$-NDPPs). However, the runtime of this approach is quadratic in $n$, making it infeasible for large-scale settings.  In this work, we develop a scalable MCMC sampling algorithm for $k$-NDPPs with low-rank kernels, thus enabling runtime that is sublinear in $n$.  Our method is based on a state-of-the-art NDPP rejection sampling algorithm, which we enhance with a novel approach for efficiently constructing the proposal distribution.  Furthermore, we extend our scalable $k$-NDPP sampling algorithm to NDPPs without size constraints.  Our resulting sampling method has polynomial time complexity in the rank of the kernel, while the existing approach has runtime that is exponential in the rank.  With both a theoretical analysis and experiments on real-world datasets, we verify that our scalable approximate sampling algorithms are orders of magnitude faster than existing sampling approaches for $k$-NDPPs and NDPPs.

> Oral: Independent Policy Gradient for Large-Scale Markov Potential Games: Sharper Rates, Function Approximation, and Game-Agnostic Convergence 
> 
> Authors: Dongsheng Ding and Chen-Yu Wei and Mihailo Jovanovic and Kaiqing Zhang
> 
> Abstract: We study the global, non-asymptotic convergence of policy gradient methods for a class of multi-agent reinforcement learning in the infinite-horizon Markov potential game (MPG) framework. Our focus is a practical large-scale setting in which the size of state space and (or) the number of players can be very large. We exploit the structural property of the policy gradient to propose new independent policy gradient algorithms run by all players in tandem for learning a Nash equilibrium of a large-scale MPG. In the exact gradient case, we show that our algorithm finds an $\epsilon$-Nash equilibrium in $O(1/\epsilon^2)$ iteration complexity. Such iteration complexity does not explicitly depend on the state space size. In the sample-based case, our algorithm works in the function approximation setting, and we prove $O(1/\epsilon^5)$ sample complexity bound in a potentially infinitely large state space. This appears to be the first result for learning MPGs with function approximation. Moreover, we identify a class of independent policy gradient algorithms that enjoy convergence for both zero-sum Markov games and Markov cooperative games, a special case of MPGs,  while the players are oblivious to the types of games being played, i.e., {\it game-agnostic}. This finding sheds light on an open question in the literature on the existence of such an algorithm.  Finally, we provide experimental    results to corroborate the merits of our theoretical developments.

> Oral: Overcoming Oscillations in Quantization-Aware Training
> 
> Authors: Markus Nagel and   and Yelysei Bondarenko and Tijmen Blankevoort
> 
> Abstract: When training neural networks with simulated quantization, we observe that quantized weights can, rather unexpectedly, oscillate between two grid-points. The importance of this effect and its impact on quantization-aware training are not well-understood or investigated in literature. In this paper, we delve deeper into the phenomenon of weight oscillations and show that it can lead to a significant accuracy degradation due to the wrong estimation of the batch-normalization statistics after training and causes increased noise during optimization. These effects are particularly pronounced in low-bit (≤ 4-bits) quantization of efficient networks with depth-wise separable layers, such as MobileNets and EfficientNets. In our analysis we investigate several previously proposed quantization-aware-training (QAT) algorithms and show that most of these are unable to overcome oscillations. Finally, we propose two new QAT algorithms to overcome oscillations during training: oscillation dampening and iterative weight freezing. These new algorithms outperform most existing QAT algorithms on several efficient neural network architectures for 3 and 4-bit quantization.


> Oral: FEDNEST: Federated Bilevel Optimization
> 
> Authors: Davoud Ataee Tarzanagh and Mingchen Li and Christos Thrampoulidis and Samet Oymak
> 
> Abstract: Standard federated optimization methods such as Federated Averaging are being successfully applied to solve stochastic problems with a single-level structure. However, many contemporary ML problems -- including adversarial robustness, hyperparameter tuning, actor-critic -- fall under nested bilevel programming that subsumes compositional and min-max optimization. In this work, we propose FEDNEST: A federated alternating stochastic gradient method to address general nested problems. We establish provable convergence rates for FEDNEST in the presence of heterogeneous data and introduce variations for specific instances. FEDNEST introduces multiple innovations including federated hypergradient computation and variance reduction to address inner-level heterogeneity. We complement our theory with experiments on hyperparameter tuning that demonstrate the benefits of our method in practice.  


> Oral: UniRank: Unimodal Bandit Algorithms for Online Ranking
> 
> Authors: Camille-Sovanneary GAUTHIER and Romaric Gaudel and Elisa Fromont
> 
> Abstract: We tackle, in the multiple-play bandit setting, the online ranking problem of assigning L items to K predefined positions on a web page in order to maximize the number of user clicks. We propose a generic algorithm, UniRank, that tackles state-of-the-art click models.  The regret bound of this algorithm is a direct consequence of the pseudo-unimodality property of the bandit setting with respect to a graph where nodes are ordered sets of indistinguishable items.  The main contribution of UniRank is its O(L/∆ logT) regret for T consecutive assignments, where ∆ relates to the reward-gap between two items.  This regret bound is based on the usually implicit condition that two items may not have the same attractiveness. Experiments against state-of-the-art learning algorithms specialized or not for different click models, show that our method has better regret performance than other generic algorithms on two real life datasets.


> Oral: Training Characteristic Functions with Reinforcement Learning: XAI-methods play Connect Four
> 
> Authors: Stephan Wäldchen and Sebastian Pokutta and Felix Huber
> 
> Abstract: Characteristic functions (from cooperative game theory) are able to evaluate partial inputs and form the basis for attribution methods like Shapley values. These attribution methods allow us to measure how important each input component is for the function output---one of the goals of explainable AI (XAI).Given a standard classifier function, it is unclear how partial input should be realised.Instead, most XAI-methods for black-box classifiers like neural networks consider counterfactual inputs that generally lie off-manifold, which makes them hard to evaluate and easy to manipulate.We propose a setup to directly train characteristic functions in the form of neural networks to play simple two-player games. We apply this to the game of Connect Four by randomly hiding colour information from our agents during training. This has three advantages for comparing XAI-methods: It alleviates the ambiguity about how to realise partial input, makes off-manifold evaluation unnecessary and allows us to compare the methods by letting them play against each other.


> Oral: Align-RUDDER: Learning From Few Demonstrations by Reward Redistribution
> 
> Authors: Vihang Patil and Markus Hofmarcher and Marius-Constantin Dinu and Matthias Dorfer and Patrick Blies and Johannes Brandstetter and Jose Antonio Arjona-Medina and Sepp Hochreiter
> 
> Abstract: Reinforcement Learning algorithms require a large number of samples to solve complex tasks with sparse and delayed rewards. Complex tasks are often hierarchically composed of sub-tasks.Solving a sub-task increases the return expectation and leads to a step in the Q-function. RUDDER identifies these steps and then redistributes reward to them, thus immediately giving reward if sub-tasks are solved. Since the delay of rewards is reduced, learning is considerably sped up.However, for complex tasks, current exploration strategies struggle with discovering episodes with high rewards.Therefore, we assume that episodes with high rewards are given as demonstrations and do not have to be discovered by exploration.Unfortunately, the number of demonstrations is typically small and RUDDER's LSTM as a deep learning model does not learn well on these few training samples.Hence, we introduce Align-RUDDER, which is RUDDER with two major modifications. First, Align-RUDDER assumes that episodes with high rewards are given as demonstrations, replacing RUDDER’s safe exploration and lessons replay buffer.Second, we substitute RUDDER’s LSTM model by a profile model that is obtained from multiple sequence alignment of demonstrations. Profile models can be constructed from as few as two demonstrations.Align-RUDDER uses reward redistribution to speed up learning by reducing the delay of rewards. Align-RUDDER outperforms competitors on complex artificial tasks with delayed rewards and few demonstrations.On the MineCraft ObtainDiamond task, Align-RUDDER is able to mine a diamond, though not frequently. 


> Oral: Stochastic Deep Networks with Linear Competing Units for Model-Agnostic Meta-Learning
> 
> Authors: Konstantinos Kalais and Sotirios Chatzis
> 
> Abstract: This work addresses meta-learning (ML) by considering deep networks with stochastic local winner-takes-all (LWTA) activations. This type of network units result in sparse representations from each model layer, as the units are organized into blocks where only one unit generates a non-zero output. The main operating principle of the introduced units lies on stochastic principles, as the network performs posterior sampling over competing units to select the winner. Therefore, the proposed networks are explicitly designed to extract input data representations of sparse stochastic nature, as opposed to the currently standard deterministic representation paradigm. Our approach produces state-of-the-art predictive accuracy on few-shot image classification and regression experiments, as well as reduced predictive error on an active learning setting; these improvements come with an immensely reduced computational cost.


> Oral: Bayesian Model Selection, the Marginal Likelihood, and Generalization
> 
> Authors: Sanae Lotfi and Pavel Izmailov and Gregory Benton and Micah Goldblum and Andrew Wilson
> 
> Abstract: How do we compare between hypotheses that are entirely consistent with observations? The marginal likelihood (Bayesian evidence), which represents the probability of generating our observations from a prior, provides a distinctive approach to this foundational question, automatically encoding Occam's razor. Although it has been observed that the marginal likelihood can overfit and is sensitive to prior assumptions, its limitations for hyperparameter learning and discrete model comparison have not been thoroughly investigated. We first revisit the appealing properties of the marginal likelihood for learning constraints and hypothesis testing. We then highlight the conceptual and practical issues in using the marginal likelihood as a proxy for generalization. Namely, we show how marginal likelihood can be negatively correlated with generalization, with implications for neural architecture search, and can lead to both underfitting and overfitting in hyperparameter learning. We provide a partial remedy through a conditional marginal likelihood, which we show is more aligned with generalization, and practically valuable for large-scale hyperparameter learning, such as in deep kernel learning.


> Poster: Selling Data To a Machine Learner: Pricing via Costly Signaling
> 
> Authors: Junjie Chen and Minming Li and Haifeng Xu
> 
> Abstract: We consider a new problem of selling data to a machine learner  who looks to purchase   data to train his machine learning model. A key challenge in this setup is that neither the seller nor the machine learner knows the true quality of data. When designing a revenue-maximizing mechanism, a data seller  faces the tradeoff between the cost and precision of data quality estimation. To address this challenge, we study a natural class of  mechanisms that price data via  costly signaling. Motivated by the assumption of  i.i.d. data points as in classic machine learning models, we first consider selling homogeneous data and derive an optimal selling mechanism. We then turn to the sale of heterogeneous data,  motivated by the sale of multiple data sets, and show that 1) on the negative side, it is NP-hard to approximate the optimal mechanism within a  constant ratio e/(e+1) + o(1); while 2) on the positive side, there is a 1/k-approximate algorithm, where k is the number of the machine learner’s private types. 


> Poster: Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing?
> 
> Authors: Keshigeyan Chandrasegaran and Ngoc-Trung Tran and Yunqing ZHAO and Ngai-Man Cheung
> 
> Abstract: This work investigates the compatibility between label smoothing (LS) and knowledge distillation (KD). Contemporary findings addressing this thesis statement take dichotomous standpoints: Muller et al. (2019) and Shen et al. (2021b). Critically, there is no effort to understand and resolve these contradictory findings, thereby creating an existential conundrum regarding the compatibility between LS and KD. The main contributions of our work are the discovery, analysis and validation of systematic diffusion as the missing concept which is instrumental in understanding and resolving these contradictory findings. This systematic diffusion essentially curtails the benefits of distilling from an LS-trained teacher, thereby rendering KD at increased temperatures ineffective. Our discovery is comprehensively supported by large-scale experiments, analyses and case studies including image classification, neural machine translation and compact student distillation tasks spanning across multiple datasets and teacher-student architectures. We further provide empirical guidelines for practitioners regarding the combined use of LS and KD. 


> Poster: A Theoretical Comparison of Graph Neural Network Extensions
> 
> Authors: Pál András Papp and Roger Wattenhofer
> 
> Abstract: We study and compare different Graph Neural Network extensions that increase the expressive power of GNNs beyond the Weisfeiler-Leman test. We focus on (i) GNNs based on higher order WL methods, (ii) GNNs that preprocess small substructures in the graph, (iii) GNNs that preprocess the graph up to a small radius, and (iv) GNNs that slightly perturb the graph to compute an embedding. We begin by presenting a simple improvement for this last extension that strictly increases the expressive power of this GNN variant. Then, as our main result, we compare the expressiveness of these extensions to each other through a series of example constructions that can be distinguished by one of the extensions, but not by another one. We also show negative examples that are particularly challenging for each of the extensions, and we prove several claims about the ability of these extensions to count cliques and cycles in the graph.


> Poster: SpaceMAP: Visualizing High-Dimensional Data by Space Expansion
> 
> Authors: Xinrui Zu and Qian Tao
> 
> Abstract: Dimensionality reduction (DR) of high-dimensional data is of theoretical and practical interest in machine learning. However, there exists an intriguing, non-intuitive discrepancy between the geometry of high- and low-dimensional space. We look into this discrepancy and propose a novel visualization method called Space-based Manifold Approximation and Projection (SpaceMAP). Our method establishes an analytical transformation on distance metrics between spaces to address the ``crowding problem" in DR. With the proposed equivalent extended distance (EED) theory, we are able to match the capacity of high- and low-dimensional space in a principled manner. To handle complex data with different manifold properties, we propose the hierarchical manifold approximation to model the similarity function in a dataset-specific manner. We evaluated SpaceMAP on a range of synthetic and real datasets with varying manifold properties, and demonstrated its excellent performance in comparison with classical and state-of-the-art DR methods. In particular, the concept of space expansion provides a generic framework for understanding nonlinear DR methods including t-distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP).


> Poster: Learning Stochastic Shortest Path with Linear Function Approximation
> 
> Authors: Yifei Min and Jiafan He and Tianhao Wang and Quanquan Gu
> 
> Abstract: We study the stochastic shortest path (SSP) problem in reinforcement learning with linear function approximation, where the transition kernel is represented as a linear mixture of unknown models. We call this class of SSP problems as linear mixture SSP. We propose a novel algorithm for learning the linear mixture SSP, which can attain a $\tilde{\mathcal{O}}(d{B_{\star}}^{1.5}\sqrt{K/c_{\min}})$ regret. Here $K$ is the number of episodes, $d$ is the dimension of the feature mapping in the mixture model, $B_{\star}$ bounds the expected cumulative cost of the optimal policy, and $c_{\min}>0$ is the lower bound of the cost function. Our algorithm also applies to the case when $c_{\min} = 0$, where a $\tilde{\mathcal{O}}(K^{2/3})$ regret is guaranteed. To the best of our knowledge, this is the first algorithm with a sublinear regret guarantee for learning linear mixture SSP. In complement to the regret upper bounds, we also prove a lower bound of $\Omega(d {B_{\star}} \sqrt{K})$, which nearly matches our upper bound.

> Poster: Residual-based Sampling for Online Outlier Robust PCA
> 
> Authors: Tianhao Zhu and Jie Shen
> 
> Abstract: Outlier robust principal component analysis (ORPCA) has been broadly applied in scientific discovery in the last decades. In this paper, we study online ORPCA, an important variant of ORPCA where the data points arrive in a sequential manner and the goal is to recover the underlying subspace of the clean data with one pass of the data. Our main contribution is the first provable algorithm that enjoys a comparable recovery guarantee to the best known batch algorithm, while significantly improving upon the state-of-the-art online ORPCA algorithms. The core technique is a robust version of the residual norm which, informally speaking, leverages not only the importance of a data point, but also how likely it behaves as an outlier.


> Poster: Hermite Polynomial Features for Private Data Generation
> 
> Authors: Margarita Vinaroz and Mohammad-Amin Charusaie and Frederik Harder and Kamil Adamczewski and Mi Jung Park
> 
> Abstract: Kernel mean embedding is a useful tool to compare probability measures. Despite its usefulness, kernel mean embedding considers infinite-dimensional features, which are challenging to handle in the context of differentially private datageneration. A recent work, DP-MERF (Harder et al., 2021), proposes to approximate the kernel mean embedding of data distribution using finite-dimensional random features, which yields an analytically tractable sensitivity of approximate kernel mean embedding. However, the requirednumber of random features in DP-MERF is excessively high, often ten thousand to a hundred thousand, which worsens the sensitivity of the approximate kernel mean embedding. To improve the sensitivity, we propose to replace random features with Hermite polynomial features. Unlike the random features, the Hermite polynomial features are ordered, where the features at the low orders contain more information on the distribution than those at the high orders. Hence, a relatively low order of Hermite polynomial features can more accurately approximate the mean embedding of the data distribution compared to a significantly higher number of random features. As a result, the Hermite polynomial features helpus to improve the privacy-accuracy trade-off compared to DP-MERF, as demonstrated on several heterogeneous tabular datasets, as well as severalimage benchmark datasets.


> Poster: Consensus Multiplicative Weights Update: Learning to Learn using Projector-based Game Signatures
> 
> Authors: Nelson Vadori and Rahul Savani and Thomas Spooner and Sumitra Ganesh
> 
> Abstract: Cheung and Piliouras (2020) recently showed that two variants of the Multiplicative Weights Update method - OMWU and MWU - display opposite convergence properties depending on whether the game is zero-sum or cooperative. Inspired by this work and the recent literature on learning to optimize for single functions, we introduce a new framework for learning last-iterate convergence to Nash Equilibria in games, where the update rule's coefficients (learning rates) along a trajectory are learnt by a reinforcement learning policy that is conditioned on the nature of the game: \textit{the game signature}. We construct the latter using a new decomposition of two-player games into eight components corresponding to commutative projection operators, generalizing and unifying recent game concepts studied in the literature. We compare the performance of various update rules when their coefficients are learnt, and show that the RL policy is able to exploit the game signature across a wide range of game types. In doing so, we introduce CMWU, a new algorithm that extends consensus optimization to the constrained case, has local convergence guarantees for zero-sum bimatrix games, and show that it enjoys competitive performance on both zero-sum games with constant coefficients and across a spectrum of games when its coefficients are learnt.


> Poster: Volatility Based Kernels and Moving Average Means for Accurate Forecasting with Gaussian Processes
> 
> Authors: Gregory Benton and Wesley Maddox and Andrew Wilson
> 
> Abstract: A broad class of stochastic volatility models are defined by systems of stochastic differential equations, and while these models have seen widespread success in domains such as finance and statistical climatology, they typically lack an ability to condition on historical data to produce a true posterior distribution. To address this fundamental limitation, we show how to re-cast a class of stochastic volatility models as a hierarchical Gaussian process (GP) model with specialized covariance functions. This GP model retains the inductive biases of the stochastic volatility model while providing the posterior predictive distribution given by GP inference. Within this framework, we take inspiration from well studied domains to introduce a new class of models, Volt and Magpie, that significantly outperform baselines in stock and wind speed forecasting, and naturally extend to the multitask setting.


> Poster: A Joint Exponential Mechanism For Differentially Private Top-$k$
> 
> Authors: Jennifer Gillenwater and Matthew Joseph and andres munoz and Monica Ribero Diaz
> 
> Abstract: We present a differentially private algorithm for releasing the sequence of $k$ elements with the highest counts from a data domain of $d$ elements. The algorithm is a "joint" instance of the exponential mechanism, and its output space consists of all $O(d^k)$ length-$k$ sequences. Our main contribution is a method to sample this exponential mechanism in time $O(dk\log(k) + d\log(d))$ and space $O(dk)$. Experiments show that this approach outperforms existing pure differential privacy methods and improves upon even approximate differential privacy methods for moderate $k$.

> Poster: Optimally Controllable Perceptual Lossy Compression
> 
> Authors: Zeyu Yan and Fei Wen and Peilin Liu
> 
> Abstract: Recent studies in lossy compression show that distortion and perceptual quality are at odds with each other, which put forward the tradeoff between distortion and perception (D-P). Intuitively, to attain different perceptual quality, different decoders have to be trained. In this paper, we present a nontrivial finding that only two decoders are sufficient for optimally achieving arbitrary (an infinite number of different) D-P tradeoff. We prove that arbitrary points of the D-P tradeoff bound can be achieved by a simple linear interpolation between the outputs of a minimum MSE decoder and a specifically constructed perfect perceptual decoder. Meanwhile, the perceptual quality (in terms of the squared Wasserstein-2 distance metric ) can be quantitatively controlled by the interpolation factor. Furthermore, to construct a perfect perceptual decoder, we propose two theoretically optimal training frameworks. The new frameworks are different from the distortion-plus-adversarial loss based heuristic framework widely used in existing methods, which are not only theoretically optimal but also can yield state-of-the-art performance in practical perceptual decoding. Finally, we validate our theoretical finding and demonstrate the superiority of our frameworks via experiments.


> Poster: Variational On-the-Fly Personalization
> 
> Authors: Kim Jangho and Jun-Tae Lee and Simyung Chang and NOJUN KWAK
> 
> Abstract: With the development of deep learning (DL) technologies, the demand for DL-based services on personal devices, such as mobile phones, also increases rapidly. In this paper, we propose a novel personalization method, Variational On-the-Fly Personalization. Compared to the conventional personalization methods that require additional fine-tuning with personal data, the proposed method only requires forwarding a handful of personal data on-the-fly. Assuming even a single personal data can convey the characteristics of a target person, we develop the variational hyper-personalizer to capture the weight distribution of layers that fits the target person. In the testing phase, the hyper-personalizer estimates the model's weights on-the-fly based on personality by forwarding only a small amount of (even a single) personal enrollment data. Hence, the proposed method can perform the personalization without any training software platform and additional cost in the edge device. In experiments, we show our approach can effectively generate reliable personalized models via forwarding (not back-propagating) a handful of samples.


> Poster: Codeformer: Learning to Translate from C to CUDA
> 
> Authors: Yuanbo Wen and Qi Guo and Qiang Fu and XiaQing Li and jianxing xu and Yanlin Tang and Yongwei Zhao and Xing Hu and Zidong Du and Ling Li and Chao Wang and Xuehai Zhou and Yunji Chen
> 
> Abstract: GPUs have become the dominant computing platforms for many applications, while programming GPUs with the widely-used CUDA parallel programming model is difficult. As sequential C code is relatively easy to obtain either from legacy repositories or by manual implementation, automatically translating C to its parallel CUDA counterpart is promising to relieve the burden of GPU programming. However, because of huge differences between the sequential C and the parallel CUDA programming model, existing approaches fail to conduct the challenging auto-parallelized program translation. In this paper, we propose a learning-based framework, i.e., Codeformer, to address this problem. We first create a large-scale dataset consisting of compute-intensive function-level monolingual corpora.We further propose using back-translation with a discriminative reranker to cope with unpaired corpora and parallel semantic conversion.Experimental results show that Codeformer outperforms state-of-the-art by 1.79, 6.09, and 9.51 in terms of BLEU, CodeBLEU, and specifically designed ParaBLEU, respectively. The CUDA code generated by Codeformer attains a speedup of up to 347x over the sequential C code, and the developer productivity is improved by at most 3.8x.


> Poster: LCANets: Lateral Competition Improves Robustness Against Corruption and Attack
> 
> Authors: Michael Teti and Juston Moore and Garrett T Kenyon and Benjamin Migliori
> 
> Abstract: Although Convolutional Neural Networks (CNNs) achieve high accuracy on image recognition tasks, they lack robustness against realistic corruptions and fail catastrophically when deliberately attacked. Previous work suggests that CNNs with representations similar to primary visual cortex (V1) are more robust to adversarial attacks on images than current adversarial defense techniques. However, these approaches require training on large-scale neural recordings or handcrafting classical neuroscientific models. Motivated by long-standing evidence that neural activity in V1 and other sensory areas is sparse, we develop a class of hybrid CNNs, called LCANets, which feature a frontend with recurrent lateral competition. We demonstrate competitive accuracy of LCANets on action recognition datasets and show that LCANets are significantly more robust to image  corruptions and adversarial  examples than both standard CNNs and adversarially-trained CNNs. Our results demonstrate that recurrent lateral competition plays a large role in forming sparse and robust V1-like representations when incorporated into CNNs.


> Poster: Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks
> 
> Authors: Litian Liang and Yaosheng Xu and Stephen Mcaleer and Dailin Hu and Alexander Ihler and Pieter Abbeel and Roy Fox
> 
> Abstract: In temporal-difference reinforcement learning algorithms, variance in value estimation can cause instability and overestimation of the maximal target value. Many algorithms have been proposed to reduce overestimation, including several recent ensemble methods, however none have shown success in sample-efficient learning through addressing estimation variance as the root cause of overestimation. In this paper, we propose MeanQ Network, a simple ensemble method that estimates target values as ensemble means. Despite its simplicity, MeanQ shows remarkable sample efficiency in experiments on the Atari Learning Environment. Importantly, we find that an ensemble of size 5 sufficiently reduces estimation variance to obviate the lagging target network, eliminating it as a source of bias and further gaining sample efficiency. We justify theoretically and empirically the design choices in MeanQ, including the necessity of independent experience sampling. On a set of 26 benchmark Atari environments, MeanQ outperforms all tested baselines, including the best available baseline, Sunrise, at 100K interaction steps in 16/26 environments, and by 68\% on average. MeanQ also outperforms Rainbow DQN at 500K steps in 21/26 environments, and by 49\% on average, and achieves average human-level performance using 200K ($\pm$ 100K) interaction steps.

> Poster: Disentangled Federated Learning for Tackling Attributes Skew via Invariant Aggregation and Diversity Transferring
> 
> Authors: Zhengquan Luo and Yunlong Wang and Zilei Wang and Zhenan Sun and Tieniu Tan
> 
> Abstract: Attributes skew hinders the current federated learning (FL) frameworks from  consistent optimization directions among the clients, which inevitably leads to performance reduction and unstable convergence. The core problems lie in that: 1) Domain-specific attributes, which are non-causal and only locally valid, are indeliberately mixed into global aggregation. 2) Two conflicting objectives, i.e., generalization and personalization, cannot be satisfied simultaneously by the one-stage optimizations of entangled attributes. To cope with these, we proposed disentangled federated learning (DFL) to disentangle the domain-specific and cross-invariant attributes into two complementary branches, which are trained by the proposed alternating local-global optimization independently. Importantly, convergence analysis proves that the FL system can be stably converged even if incomplete client models participate in the global aggregation, which greatly expands the application scope of FL. Extensive experiments verify that DFL facilitates FL with higher performance, better interpretability, and faster convergence rate, compared with SOTA FL methods on both manually synthesized and realistic attributes skew datasets.


> Poster: Decentralized Online Convex Optimization in Networked Systems
> 
> Authors: Yiheng Lin and Judy Gan and Guannan Qu and Yash Kanoria and Adam Wierman
> 
> Abstract: We study the problem of networked online convex optimization, where each agent individually decides on an action at every time step and agents cooperatively seek to minimize the total global cost over a finite horizon. The global cost is made up of three types of local costs: convex node costs, temporal interaction costs, and spatial interaction costs. In deciding their individual action at each time, an agent has access to predictions of local cost functions for the next $k$ time steps in an $r$-hop neighborhood. Our work proposes a novel online algorithm, Localized Predictive Control (LPC), which generalizes predictive control to multi-agent systems. We show that LPC achieves a competitive ratio of $1 + \tilde{O}(\rho_T^k) + \tilde{O}(\rho_S^r)$ in an adversarial setting, where $\rho_T$ and $\rho_S$ are constants in $(0, 1)$ that increase with the relative strength of temporal and spatial interaction costs, respectively. This is the first competitive ratio bound on decentralized predictive control for networked online convex optimization. Further, we show that the dependence on $k$ and $r$ in our results is near optimal by lower bounding the competitive ratio of any decentralized online algorithm.

> Poster: Variational Wasserstein gradient flow
> 
> Authors: Jiaojiao Fan and Qinsheng Zhang and Amirhossein Taghvaei and Yongxin Chen
> 
> Abstract: Wasserstein gradient flow has emerged as a promising approach to solve optimization problems over the space of probability distributions. A recent trend is to use the well-known JKO scheme in combination with input convex neural networks to numerically implement the proximal step. The most challenging step, in this setup, is to evaluate functions involving density explicitly, such as entropy, in terms of samples. This paper builds on the recent works with a slight but crucial difference: we propose to utilize a variational formulation of the objective function formulated as maximization over a parametric class of functions. Theoretically, the proposed variational formulation allows the construction of gradient flows directly for empirical distributions with a well-defined and meaningful objective function. Computationally, this approach replaces the computationally expensive step in existing methods, to handle objective functions involving density, with inner loop updates that only require a small batch of samples and scale well with the dimension. The performance and scalability of the proposed method are illustrated with the aid of several numerical experiments involving high-dimensional synthetic and real datasets.


> Poster: Scalable Deep Reinforcement Learning Algorithms for Mean Field Games
> 
> Authors: Mathieu Lauriere and Sarah Perrin and Sertan Girgin and Paul Muller and Ayush Jain and Theophile Cabannes and Georgios Piliouras and Julien Perolat and Romuald Elie and Olivier Pietquin and Matthieu Geist
> 
> Abstract: Mean Field Games (MFGs) have been introduced to efficiently approximate games with very large populations of strategic agents. Recently, the question of learning equilibria in MFGs has gained momentum, particularly using model-free reinforcement learning (RL) methods. One limiting factor to further scale up using RL is that existing algorithms to solve MFGs require the mixing of approximated quantities (such as strategies or q-values). This is non-trivial in the case of non-linear function approximation (e.g. neural networks). We propose two methods to address this shortcoming. One learns a mixed strategy from distillation of historical data into a neural network and is applied to the Fictitious Play algorithm. The other is an online mixing method based on regularization that does not require memorizing historical data or previous estimates. It is used to extend Online Mirror Descent. We demonstrate numerically that these methods efficiently enable the use of Deep RL algorithms to solve various MFGs. In addition, we show that, thanks to generalization, the resulting algorithms outperform their SotA counterparts.


> Poster: Difference Advantage Estimation for Multi-Agent Policy Gradients
> 
> Authors: 岳珩 李 and Guangming Xie and Zongqing Lu
> 
> Abstract: Multi-agent policy gradient methods in centralized training with decentralized execution recently witnessed many progresses. During centralized training, multi-agent credit assignment is crucial, which can substantially promote learning performance. However, explicit multi-agent credit assignment in multi-agent policy gradient methods still receives less attention. In this paper, we investigate multi-agent credit assignment induced by reward shaping and provide a theoretical understanding in terms of its credit assignment and policy bias. Based on this, we propose an exponentially weighted advantage estimator, which is analogous to GAE, to enable multi-agent credit assignment while allowing the tradeoff with policy bias. Empirical results show that our approach can successfully perform effective multi-agent credit assignment, and thus substantially outperforms other advantage estimators.


> Poster: Estimating and Penalizing Induced Preference Shifts in Recommender Systems
> 
> Authors: Micah Carroll and Dylan Hadfield-Menell and Stuart Russell and Anca Dragan
> 
> Abstract: The content that a recommender system (RS) shows to users will influence them. Therefore, when choosing which recommender to deploy, one is implicitly also choosing to induce specific internal states in users. Even more, systems trained via long-horizon optimization will have direct incentives to manipulate users, e.g. shift their preferences so they are easier to satisfy. In this work we focus on induced preference shifts in users. We argue that -- before deployment -- system designers should: estimate the shifts a recommender would induce; evaluate whether such shifts would be undesirable; and even actively optimize to avoid problematic shifts. These steps involve two challenging ingredients: estimation requires anticipating how hypothetical policies would influence user preferences if deployed -- we do this by using historical user interaction data to train predictive user model which implicitly contains their preference dynamics; evaluation and optimization additionally require metrics to assess whether such influences are manipulative or otherwise unwanted -- we use the notion of “safe shifts”, that define a trust region within which behavior is safe. In simulated experiments, we show that our learned preference dynamics model is effective in estimating user preferences and how they would respond to new recommenders. Additionally, we show that recommenders that optimize for staying in the trust region can avoid manipulative behaviors while still generating engagement.


> Poster: Fast Provably Robust Decision Trees and Boosting
> 
> Authors: Junqi Guo and Ming-Zhuo Teng and Wei Gao and Zhi-Hua Zhou
> 
> Abstract: Learning with adversarial robustness has been a challenge in contemporary machine learning, and recent years have witnessed increasing attention on robust decision trees and ensembles, mostly working with high computational complexity or without guarantees of provable robustness. This work proposes the Fast Provably Robust Decision Tree (FPRDT) with the smallest computational complexity O(n log n), a tradeoff between global and local optimizations over the adversarial 0/1 loss. We further develop the Provably Robust AdaBoost (PRAdaBoost) according to our robust decision trees, and present convergence analysis for training adversarial 0/1 loss. We conduct extensive experiments to support our approaches; in particular, our approaches are superior to those unprovably robust methods,  and achieve better or comparable performance to those provably robust methods yet with the smallest running time.


> Poster: Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets
> 
> Authors: Lily Zhang and Veronica Tozzo and John Higgins and Rajesh Ranganath
> 
> Abstract: Permutation invariant neural networks are a promising tool for predictive modeling of set data. We show, however, that existing architectures struggle to perform well when they are deep. In this work, we mathematically and empirically analyze normalization layers and residual connections in the context of deep permutation invariant neural networks. We develop set norm, a normalization tailored for sets, and introduce the ``clean path principle'' for equivariant residual connections alongside a novel benefit of such connections, the reduction of information loss.  Based on our analysis, we propose Deep Sets++ and Set Transformer++, deep models that reach comparable or better performance than their original counterparts on a diverse suite of tasks. We additionally introduce Flow-RBC, a new single-cell dataset and real-world application of permutation invariant prediction. We open-source our data and code here: link-omitted-for-anonymity.


> Poster: RUMs from Head-to-Head Contests
> 
> Authors: Matteo Almanza and Flavio Chierichetti and Ravi Kumar and Alessandro Panconesi and Andrew Tomkins
> 
> Abstract: Random utility models (RUMs) encode the likelihood that a particular item will be selected from a slate of competing items. RUMs are well-studied objects in both discrete choice theory and, more recently, in the learning community, as they encode a fairly broad notion of rational user behavior. In this paper, we focus on slates of size two representing head-to-head competitions. Given a tournament matrix $M$ such that $M_{i,j}$ is the probability that item $i$ will be selected from $\{i, j\}$, we consider the problem of finding the RUM that most closely reproduces $M$.  For this problem we obtain a polynomial-time algorithm returning a RUM that approximately minimizes the average error over the pairs. Our experiments show that RUMs can {\em perfectly} represent many of the tournament matrices that have been considered in the literature; in fact, the maximum average error induced by RUMs on the matrices we considered is negligible ($\approx 0.001$). We also show that RUMs are competitive, on prediction tasks, with previous approaches.

> Poster: Regret Minimization with Performative Feedback
> 
> Authors: Meena Jagadeesan and Tijana Zrnic and Celestine Mendler-Dünner
> 
> Abstract: In performative prediction, the deployment of a predictive model triggers a shift in the data distribution. As these shifts are typically unknown ahead of time, the learner needs to deploy a model to get feedback about the distribution it induces. We study the problem of finding near-optimal models under performativity while maintaining low regret. On the surface, this problem might seem equivalent to a bandit problem. However, it exhibits a fundamentally richer feedback structure that we refer to as performative feedback: after every deployment, the learner receives samples from the shifted distribution rather than bandit feedback about the reward. Our main contribution is regret bounds that scale only with the complexity of the distribution shifts and not that of the reward function. The key algorithmic idea is careful exploration of the distribution shifts that informs a novel construction of confidence bounds on the risk of unexplored models. The construction only relies on smoothness of the shifts and does not assume convexity. More broadly, our work establishes a conceptual approach for leveraging tools from the bandits literature for the purpose of regret minimization with performative feedback.


> Poster: Low-Complexity Deep Convolutional Neural Networks on Fully Homomorphic Encryption Using Multiplexed Parallel Convolutions
> 
> Authors: Eunsang Lee and Joon-Woo Lee and Junghyun Lee and Young-Sik KIM and Yongjune Kim and Jong-Seon No and Woosuk Choi
> 
> Abstract: Recently, the standard ResNet-20 network was successfully implemented on residue number system variant Cheon-Kim-Kim-Song (RNS-CKKS) scheme using bootstrapping, but the implementation lacks practicality due to high latency and low security level. To improve the performance, we first minimize total bootstrapping runtime using multiplexed parallel convolution that collects sparse output data for multiple channels compactly. We also propose the imaginary-removing bootstrapping to prevent the deep neural networks from catastrophic divergence during approximate ReLU operations. In addition, we optimize level consumptions and use lighter and tighter parameters. Simulation results show that we have 4.67x lower inference latency and 134x less amortized runtime (runtime per image) for ResNet-20 compared to the state-of-the-art previous work, and we achieve standard 128-bit security. Furthermore, we successfully implement ResNet-110 with high accuracy on the RNS-CKKS scheme for the first time. 


> Poster: DynaMixer: A Vision MLP Architecture with Dynamic Mixing
> 
> Authors: Ziyu Wang and Wenhao Jiang and Yiming Zhu and Li Yuan and Yibing Song and Wei Liu
> 
> Abstract: Recently, MLP-like vision models have achieved promising performances on mainstream visual recognition tasks. In contrast with vision transformers and CNNs, the success of MLP-like models proves that simple information fusion operations among tokens and channels can yield a good representation power for deep recognition models. However, existing MLP-like models fuse tokens through static fusion operations, lacking adaptability to the contents of the tokens to be mixed. Thus, customary information fusion procedures are not effective enough. To this end, this paper presents an efficient MLP-like network architecture, dubbed DynaMixer, resorting to dynamic information fusion. Critically, we propose a procedure, on which the DynaMixer model relies, to dynamically generate mixing matrices by leveraging the contents of all the tokens to be mixed. To reduce the time complexity and improve the robustness, a dimensionality reduction technique and a multi-segment fusion mechanism are adopted. Our proposed DynaMixer model (97M parameters) achieves 84.3\% top-1 accuracy on the ImageNet-1K dataset without extra training data, performing favorably against the state-of-the-art vision MLP models. When the number of parameters is reduced to 26M, it still achieves 82.7\% top-1 accuracy,  surpassing the existing MLP-like models with a similar capacity. The implementation of DynaMixer will be made available to the public.


> Poster: On Learning Mixture of Linear Regressions in the Non-Realizable Setting
> 
> Authors: Soumyabrata Pal and Arya Mazumdar and Rajat Sen and Avishek Ghosh
> 
> Abstract: While mixture of linear regressions (MLR) is a well-studied topic, prior works usually do not  analyze such  models for prediction error.  In fact, \emph{prediction} and \emph{loss} are not well-defined in the context of mixtures. In this paper, first we show that MLR can be used for prediction where instead of predicting a label, the model predicts a list of values (also known as \emph{list-decoding}). The list size is equal to the number of components in the mixture, and the loss function is defined to be minimum among the losses resulted by all the component models. We show that with this definition, a solution of the empirical risk minimization (ERM) achieves small probability of prediction error. This begs for an algorithm to minimize the empirical risk for MLR, which is known to be computationally hard. Prior algorithmic works in MLR focus on the \emph{realizable} setting, i.e., recovery of parameters when data is probabilistically generated by a mixed linear (noisy) model. In this paper we show that a version of the popular expectation minimization (EM) algorithm finds out the best fit lines in a dataset even when a realizable model is not assumed, under some regularity conditions on the dataset and the initial points, and thereby provides a solution for the ERM. We further provide an algorithm that runs in polynomial time in the number of datapoints, and recovers a good approximation of the best fit lines. The two algorithms are experimentally compared.


> Poster: Learning Efficient and Robust Ordinary Differential Equations via Invertible Neural Networks
> 
> Authors: Weiming Zhi and Tin Lai and Lionel Ott and Edwin V Bonilla and Fabio Ramos
> 
> Abstract: Advances in differentiable numerical integrators have enabled the use of gradient descent techniques to learn ordinary differential equations (ODEs), where a flexible function approximator (often a neural network) is used to estimate the system dynamics, given as a time derivative. However, these integrators can be unsatisfactorily slow and unstable when learning systems of ODEs from long sequences. We propose to learn an ODE of interest from data by viewing its dynamics as a vector field related to another base vector field via a diffeomorphism (i.e., a differentiable bijection), represented by an invertible neural network (INN). By learning both the INN and the dynamics of the base ODE, we provide an avenue to offload some of the complexity in modelling the dynamics directly on to the INN. Consequently, by restricting the base ODE to be amenable to integration, we can speed up and improve the robustness of integrating trajectories from the learned system. We demonstrate the efficacy of our method in training and evaluating benchmark ODE systems, as well as within continuous-depth neural networks models. We show that our approach attains speed-ups of up to two orders of magnitude when integrating learned ODEs.


> Poster: Improving and Assessing Anomaly Detectors for Large-Scale Settings
> 
> Authors: Dan Hendrycks and Steven Basart and Mantas Mazeika and Andy Zou and joseph kwon and Mohammadreza Mostajabi and Jacob Steinhardt
> 
> Abstract: Detecting out-of-distribution examples is important for safety-critical machine learning applications such as detecting novel biological phenomena and self-driving cars. However, existing research mainly focuses on simple small-scale settings. To set the stage for more realistic out-of-distribution detection, we depart from small-scale settings and explore large-scale multiclass and multi-label settings with high-resolution images and thousands of classes. To make future work in real-world settings possible, we create new benchmarks for three large-scale settings. To test ImageNet multiclass anomaly detectors, we introduce a new dataset of anomalous species. We leverage ImageNet-22K to evaluate PASCAL VOC and COCO multilabel anomaly detectors. Third, we introduce a new benchmark for anomaly segmentation by introducing a segmentation benchmark with road anomalies. We conduct extensive experiments in these more realistic settings for out-of-distribution detection and find that a surprisingly simple detector based on the maximum logit outperforms prior methods in all the large-scale multi-class, multi-label, and segmentation tasks, establishing a simple new baseline for future work.


> Poster: TPC: Transformation-Specific Smoothing for Point Cloud Models
> 
> Authors: Wenda Chu and Linyi Li and Bo Li
> 
> Abstract: Point cloud models with neural network architectures have achieved great success and been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable against adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into two categories: composable (e.g., rotation) and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for both categories. We then specify unique certification protocols for a range of specific semantic transformations and derive strong robustness guarantees. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within ±20°) from 20.3% to 83.8%.


> Poster: Discovering Generalizable Spatial Goal Representations via Graph-based Active Reward Learning
> 
> Authors: Aviv Netanyahu and Tianmin Shu and Josh Tenenbaum and Pulkit Agrawal
> 
> Abstract: In this work, we consider one-shot imitation learning for object rearrangement tasks, where an AI agent needs to watch a single expert demonstration and learn to perform the same task in different environments. To achieve a strong generalization, the AI agent must infer the spatial goal specification for the task. However, there can be multiple goal specifications that fit the given demonstration. To address this, we propose a reward learning approach, Graph-based Equivalence Mappings (GEM), that can discover spatial goal representations that are aligned with the intended goal specification, enabling successful generalization in unseen environments. Specifically, GEM represents a spatial goal specification by a reward function conditioned on i) a graph indicating important spatial relationships between objects and ii) state equivalence mappings for each edge in the graph indicating invariant properties of the corresponding relationship. GEM combines inverse reinforcement learning and active reward learning to efficiently improve the reward function by utilizing the graph structure and domain randomization enabled by the equivalence mappings. We conducted experiments with simulated oracles and with human subjects. The results show that GEM can drastically improve the generalizability of the learned goal representations over strong baselines.


> Poster: Comprehensive Analysis of Negative Sampling in Knowledge Graph Representation Learning
> 
> Authors: Hidetaka Kamigaito and Katsuhiko Hayashi
> 
> Abstract: Negative Sampling (NS) loss plays an important role in learning Knowledge Graph Embedding (KGE) to handle many entities. However, the performance of KGE will degrade if we do not properly choose hyperparameters such as the margin term and the number of negative samples in the NS loss. Currently, empirical hyperparameter tuning addresses this problem at the cost of computational time. In this paper, to deal with this problem, we theoretically analyze the NS loss to assist hyperparameter tuning and understand the better use of the NS loss in KGE learning. Our theoretical analysis shows that scoring methods with restricted value ranges, such as TransE, RotatE, and HAKE, require appropriate adjustment of the margin term or the number of negative samples different from that without restricted value ranges, such as ComplEx and DistMult. In addition, we also propose subsampling methods specialized for the NS loss in KGE studied from the theoretical aspect. Our empirical analysis on FB15k-237, WN18RR, and YAGO3-10 showed that the results of actually trained models are along with our theoretical findings. We also confirmed the performance improvement by using our proposed subsampling methods.


> Poster: Safe Exploration for Efficient Policy Evaluation and Comparison
> 
> Authors: Runzhe Wan and Branislav Kveton and Rui Song
> 
> Abstract: High-quality data plays a central role in ensuring the accuracy of policy evaluation. This paper initiates the study of efficient and safe data collection for bandit policy evaluation. We formulate the problem and investigate its several representative variants. For each variant, we analyze its statistical properties, derive the corresponding exploration policy, and design an efficient algorithm for computing it. Both theoretical analysis and experiments support the usefulness of the proposed methods. 


> Poster: Training Your Sparse Neural Network Better with Any Mask
> 
> Authors:   and Haoyu Ma and Tianlong Chen and Ying Ding and Zhangyang Wang
> 
> Abstract: Pruning large neural networks to create high-quality, independently trainable sparse masks, which can maintain similar performance to their dense counterparts, is very desirable due to the reduced space and time complexity. As research effort is focused on increasingly sophisticated pruning methods that leads to sparse subnetworks trainable from the scratch, we argue for an orthogonal, under-explored theme: improving training techniques for pruned sub-networks, i.e. sparse training. Apart from the popular belief that only the quality of sparse masks matters for sparse training, in this paper we demonstrate an alternative opportunity: one can carefully customize the sparse training techniques to deviate from the default dense network training protocols, consisting of introducing ``ghost" neurons and skip connections at the early stage of training, and strategically modifying the initialization as well as labels. Our new sparse training recipe is generally applicable to improving training from scratch with various sparse masks. By adopting our newly curated techniques, we demonstrate significant performance gains across various popular datasets (CIFAR-10,  CIFAR-100,  TinyImageNet), architectures (ResNet-18/32/104, Vgg16, MobileNet), and sparse mask options (lottery ticket, SNIP/GRASP, SynFlow, or even randomly pruning), compared to the default training protocols, especially at high sparsity levels. Codes will be publicly available.


> Poster: Learning Stable Classifiers by Transferring Unstable Features
> 
> Authors: Yujia Bao and Shiyu Chang and Regina Barzilay
> 
> Abstract: While unbiased machine learning models are essential for many applications, bias is a human-defined concept that can vary across tasks. Given only input-label pairs, algorithms may lack sufficient information to distinguish stable (causal) features from unstable (spurious) features. However, related tasks often share similar biases -- an observation we may leverage to develop stable classifiers in the transfer setting. In this work, we explicitly inform the target classifier about unstable features in the source tasks. Specifically, we derive a representation that encodes the unstable features by contrasting different data environments in the source task. We achieve robustness by clustering data of the target task according to this representation and minimizing the worst-case risk across these clusters. We evaluate our method on both text and image classifications. Empirical results demonstrate that our algorithm is able to maintain robustness on the target task for both synthetically generated environments and real-world environments. Our code will be available.


> Poster: GraphFM: Improving Large-Scale GNN Training via Feature Momentum
> 
> Authors: Haiyang Yu and Limei Wang and Bokun Wang and Meng Liu and Tianbao Yang and Shuiwang Ji
> 
> Abstract: Training of graph neural networks (GNNs) for large-scale node classification is challenging. A key difficulty lies in obtaining accurate hidden node representations while avoiding the neighborhood explosion problem. Here, we propose a new technique, named as feature momentum (FM), that uses a momentum step to incorporate historical embeddings when updating feature representations. We develop two specific algorithms, known as GraphFM-IB and GraphFM-OB, that consider in-batch and out-of-batch data, respectively.GraphFM-IB applies FM to in-batch sampled data, while GraphFM-OB applies FM to out-of-batch data that are 1-hop neighborhood of in-batch data.We provide a rigorous convergence analysis for GraphFM-IB and a theoretical justification of GraphFM-OB for the estimation error of feature embeddings. Empirically, we observe that GraphFM-IB can effectively address the neighborhood explosion problem of existing methods. In addition, GraphFM-OB achieves promising performance on multiple large-scale graph datasets.


> Poster: AdaGrad Avoids Saddle Points
> 
> Authors: Kimon Antonakopoulos and Panayotis Mertikopoulos and Georgios Piliouras and Xiao Wang
> 
> Abstract: Adaptive first-order methods in optimization have widespread ML applications due to their ability to adapt to non-convex landscapes. However, their convergence guarantees are typically stated in terms of vanishing gradient norms, which leaves open the issue of converging to undesirable saddle points (or even local maxima). In this paper, we focus on the AdaGrad family of algorithms - from scalar to full-matrix preconditioning - and we examine the question of whether the method's trajectories avoid saddle points. A major challenge that arises here is that AdaGrad's step-size (or, more accurately, the method's preconditioner) evolves over time in a filtration-dependent way, i.e., as a function of all gradients observed in earlier iterations; as a result, avoidance results for methods with a constant or vanishing step-size do not apply. We resolve this challenge by combining a series of step-size stabilization arguments with a recursive representation of the AdaGrad preconditioner that allows us to employ center-stable techniques and ultimately show that the induced trajectories avoid saddle points from almost any initial condition.


> Poster: Training Discrete Deep Generative Models via Gapped Straight-Through Estimator
> 
> Authors: Ting-Han Fan and Ta-Chung Chi and Alexander Rudnicky and Peter Ramadge
> 
> Abstract: While deep generative models have succeeded in image processing, natural language processing, and reinforcement learning, training that involves discrete random variables remains challenging due to the high variance of its gradient estimation process. A common remedy adopted by most variance reduction techniques is the Monte Carlo method, which involves time-consuming resampling and multiple function evaluations. In this work, we propose the Gapped Straight-Through (GST) estimator to reduce the variance without incurring resampling overhead. Such an estimator is inspired by the essential properties of Straight-Through Gumbel-Softmax. We determine these properties and show via an ablation study that they are essential. Experiments demonstrate that the proposed GST estimator enjoys better performance compared to strong baselines on two discrete deep generative modeling tasks, MNIST-VAE and ListOps.


> Poster: Multi-slots Online Matching with High Entropy
> 
> Authors: XINGYU LU and Qintong Wu and WENLIANG ZHONG
> 
> Abstract: Online matching with high entropy, a fundamental building block that enjoys the advantages of diversity and fairness in the recommendation and advertising, is often modeled as constrained and regularized convex programming. While most existing approaches are based on the "single slot" assumption (i.e., assign one item per iteration), they cannot be directly applied to cases with multiple slots, e.g., stock-aware top-N recommendation and advertising on multiple places. Particularly, the gradient computation and allocation could be challenging under this setting.To overcome these obstacles, we develop a novel algorithm, named Online subGradient descent for Multi-slots Allocation (OG-MA), for online matching with high entropy regularizer. OG-MA uses an efficient pooling algorithm to compute the closed-form of gradient and a roulette swapping for allocation, resulting in a linear iteration-complexity. By adopting the random permutation arriving model, our algorithm attains sublinear regret compared to the hindsight optimal allocation. Extensive experiments validate the theoretical results of the proposed algorithm.


> Poster: EAT-C: Environment-Adversarial sub-Task Curriculum for Efficient Reinforcement Learning
> 
> Authors: Shuang Ao and Tianyi Zhou and Jing Jiang and Guodong Long and Xuan Song and Chengqi Zhang
> 
> Abstract: Reinforcement learning (RL) is inefficient on long-horizon tasks due to sparse rewards and its policy can be fragile to slightly perturbed environments. We address these challenges via a curriculum of tasks with coupled environments, generated by two policies trained jointly with RL: (1) a co-operative planning policy recursively decomposing a hard task into a coarse-to-fine sub-task tree; and (2) an adversarial policy modifying the environment in each sub-task. They are complementary to acquire more informative feedback for RL: (1) provides dense reward of easier sub-tasks while (2) modifies sub-tasks' environments to be more challenging and diverse. Conversely, they are trained by RL's dense feedback on sub-tasks so their generated curriculum keeps adaptive to RL's progress. The sub-task tree enables an easy-to-hard curriculum for every policy: its top-down construction gradually increases sub-tasks the planner needs to generate, while the adversarial training between the environment and RL follows a bottom-up traversal that starts from a dense sequence of easier sub-tasks allowing more frequent environment changes. We compare EAT-C with RL/planning targeting similar problems and methods with environment generators or adversarial agents. Extensive experiments on diverse tasks demonstrate the advantages of our method on improving RL's efficiency and generalization. 


> Poster: On Distribution Shift in Learning-based Bug Detectors
> 
> Authors: Jingxuan He and Luca Beurer-Kellner and Martin Vechev
> 
> Abstract: Deep learning has recently achieved initial success in program analysis tasks such as bug detection. Lacking real bugs, most existing works construct training and test data by injecting synthetic bugs into correct programs. Despite achieving high test accuracy (e.g. >90%), the resulting bug detectors are found to be surprisingly unusable in practice, i.e., <10% precision when used to scan real software repositories. In this work, we argue that this massive performance difference is caused by distribution shift, i.e., a fundamental mismatch between the real bug distribution and the synthetic bug distribution used to train and evaluate the detectors. To address this key challenge, we propose to train a bug detector in two phases, first on a synthetic bug distribution to adapt the model to the bug detection domain, and then on a real bug distribution to drive the model towards the real distribution. During these two phases, we leverage a multi-task hierarchy, focal loss, and contrastive learning to further boost performance. We evaluate our approach extensively on three widely studied bug types, for which we construct new datasets carefully designed to capture the real bug distribution. The results demonstrate that our approach is practically effective and successfully mitigates the distribution shift: our learned detectors are highly performant on both our constructed test set and the latest version of open source repositories.


> Poster: Understanding Robust Overfitting of Adversarial Training and Beyond
> 
> Authors: Yu Chaojian and Bo Han and Li Shen and Jun Yu and Chen Gong and Mingming Gong and Tongliang Liu
> 
> Abstract: Robust overfitting widely exists in adversarial training of deep networks. The exact underlying reasons for this are still not completely understood. Here, we explore the causes of robust overfitting by comparing the data distribution of non-overfit (weak adversary) and overfitted (strong adversary) adversarial training, and observe that the distribution of the adversarial data generated by weak adversary mainly contain small-loss data. However, the adversarial data generated by strong adversary is more diversely distributed on the large-loss data and the small-loss data. Given these observations, we further designed data ablation adversarial training and identify that these small-loss data that are not worthy of the adversary strength cause robust overfitting in the strong adversary mode. To relieve this issue, we propose minimum loss constrained adversarial training (MLCAT): in a minibatch, we learn large-loss data as usual, and adopt additional measures to increase the loss of the small-loss data. Technically, MLCAT hinders data fitting when they become easy to learn to prevent robust overfitting; philosophically, MLCAT reflects the spirit of turning waste into treasure and making the best use of each adversarial data; algorithmically, we designed two realizations of MLCAT, and extensive experiments demonstrate that MLCAT can eliminate robust overfitting and further boost adversarial robustness.


> Poster: A Stochastic Multi-Rate Control Framework For Modeling Distributed Optimization Algorithms
> 
> Authors: xinwei zhang and Mingyi Hong and Sairaj Dhople and Nicola Elia
> 
> Abstract: In modern machine learning systems, distributed algorithms are deployed across applications to ensure data privacy and optimal utilization of computational resources. This work offers a fresh perspective to model, analyze, and design distributed optimization algorithms through the lens of stochastic multi-rate feedback control. We show that a substantial class of distributed algorithms---including popular Gradient Tracking for decentralized learning, and FedPD and Scaffold for federated learning---can be modeled as a certain discrete-time stochastic feedback-control system, possibly with multiple sampling rates.  This key observation allows us to develop a generic framework to analyze the convergence of the entire algorithm class. It also enables one to easily add desirable features such as differential privacy guarantees, or to deal with practical settings such as partial agent participation, communication compression, and imperfect communication in algorithm design and analysis.  


> Poster: Time Is MattEr: Temporal Self-supervision for Video Transformers
> 
> Authors: Sukmin Yun and Jaehyung Kim and Dongyoon Han and Hwanjun Song and Jung-Woo Ha and Jinwoo Shin
> 
> Abstract: Understanding temporal dynamics of video is an essential aspect of learning better video representations. Thus, recent video models have been extensively explored Transformer-based architectural designs due to their capability to capture long-term dependency of input sequences. However, we found that these Video Transformers are still biased to learn spatial dynamics rather than temporal ones, and debiasing the spurious correlation is critical for their performance. Based on the observations, we design simple yet effective self-supervised tasks for video models to learn temporal dynamics better by utilizing shuffled video and its original counterpart. Specifically, our method simultaneously learns the temporal order of video frames as extra self-supervision and enforces the randomly shuffled video frames to have low-confidence outputs. Under various video action recognition tasks, we demonstrate the effectiveness of our method and its compatibility with state-of-the-art Video Transformers.


> Poster: Understanding Robust Generalization in Learning Regular Languages
> 
> Authors: Soham Dan and Osbert Bastani and Dan Roth
> 
> Abstract: A key feature of human intelligence is the ability to generalize beyond the training distribution, for instance, parsing longer sentences than seen in the past. Currently, deep neural networks struggle to generalize robustly to such shifts in the data distribution. We study robust generalization in the context of using recurrent neural networks (RNNs) to learn regular languages. We hypothesize that standard end-to-end modeling strategies cannot generalize well to systematic distribution shifts and propose a compositional strategy to address this. We compare an end-to-end strategy that maps strings to labels with a compositional strategy that predicts the structure of the deterministic finite state automaton (DFA) that accepts the regular language. We theoretically prove that the compositional strategy generalizes significantly better than the end-to-end strategy. In our experiments, we implement the compositional strategy via an auxiliary task where the goal is to predict the intermediate states visited by the DFA when parsing a string. Our empirical results support our hypothesis, showing that auxiliary tasks can enable robust generalization. Interestingly, the end-to-end RNN generalizes significantly better than the theoretical lower bound, suggesting that it is able to achieve atleast some degree of robust generalization.


> Poster: Analyzing and Mitigating Interference in Neural Architecture Search
> 
> Authors: Jin Xu and Xu Tan and Kaitao Song and Renqian Luo and Yichong Leng and Tao Qin and Tie-Yan Liu and Jian Li
> 
> Abstract: Weight sharing is a popular approach to reduce the training cost of neural architecture search (NAS) by reusing the weights of shared operators from previously trained child models. However, the rank correlation between the estimated accuracy and ground truth accuracy of those child models is low due to the interference among different child models caused by weight sharing. In this paper, we investigate the interference issue by sampling different child models and calculating the gradient similarity of shared operators, and observe that: 1) the interference on a shared operator between two child models is positively correlated with the number of different operators between them; 2) the interference is smaller when the inputs and outputs of the shared operator are more similar. Inspired by these two observations, we propose two approaches to mitigate the interference: 1) rather than randomly sampling child models for optimization, we propose a gradual modification scheme by modifying one operator between adjacent optimization steps to minimize the interference on the shared operators; 2) forcing the inputs and outputs of the operator across all child models to be similar to reduce the interference. Experiments on a BERT search space verify that mitigating interference via each of our proposed methods improves the rank correlation of super-pet and combining both methods can achieve better results. Our discovered architecture outperforms RoBERTa$_{\rm base}$ by 1.1 and 0.6 points and ELECTRA$_{\rm base}$ by 1.6 and 1.1 points on the dev and test set of GLUE benchmark. Extensive results on the BERT compression, reading comprehension and large-scale image classification tasks also demonstrate the effectiveness and generality of our proposed methods.

> Poster: Improved StyleGAN-v2 based Inversion for Out-of-Distribution Images
> 
> Authors: Rakshith Subramanyam and Vivek Narayanaswamy and Mark Naufel and Andreas Spanias and Jayaraman J. Thiagarajan
> 
> Abstract: Inverting an image onto the latent space of pre-trained generators, e.g., StyleGAN-v2, has emerged as a popular strategy to leverage strong image priors for ill-posed restoration. Several studies have showed that this approach is highly effective at inverting images similar to the data used for StyleGAN training (e.g., FFHQ faces). However, with out-of-distribution (OOD) data that the generator has not been exposed to, existing inversion techniques often produce highly sub-optimal results. In this paper, we propose SPHInX (StyleGAN with Projection Heads for Inverting X), an approach for accurately embedding OOD images onto the StyleGAN latent space. SPHInX adopts a novel training strategy that jointly optimizes: (i) a carefully designed style projection head that replaces the mapping network in StyleGAN; (ii) a content projection head; and (iii) noise latent variables in every layer. Our empirical studies with a suite of OOD data show that, in addition to producing higher quality image reconstructions over the state-of-the-art GAN inversion techniques, SPHInX is effective at conventional restoration problems such as denoising and compressed sensing while offering semantic editing capabilities. 


> Poster: Leverage Score Sampling for Tensor Product Matrices in Input Sparsity Time
> 
> Authors: David Woodruff and Amir Zandieh
> 
> Abstract: We give an input sparsity time sampling algorithm for spectrally approximating the Gram matrix corresponding to the q-fold column-wise tensor product of q matrices using a nearly optimal number of samples, improving upon all previously known methods by poly(q) factors. Furthermore, for the important special case of the q-fold self-tensoring of a dataset, which is the feature matrix of the degree-q polynomial kernel, the leading term of our method’s runtime is proportional to the size of the dataset and has no dependence on q. Previous techniques either incur a poly(q) factor slowdown in their runtime or remove the dependence on q at the expense of having sub-optimal target dimension and depend quadratically on the number of data-points in their runtime. Our sampling technique relies on a collection of q partially correlated random projections which can be simultaneously applied to a dataset X in total time that only depends on the size of X, and at the same time their q-fold Kronecker product acts as a near-isometry for any fixed vector in the column span of $X^{\otimes q}$. We show that our sampling methods generalize to other classes of kernels beyond polynomial, such as Gaussian and Neural Tangent kernels.

> Poster: Choosing Answers in Epsilon-Best-Answer Identification for Linear Bandits
> 
> Authors: Marc Jourdan and Rémy Degenne
> 
> Abstract: In pure-exploration problems, information is gathered sequentially to answer a question on the stochastic environment.While best-arm identification for linear bandits has been extensively studied in recent years, few works have been dedicated to identifying one arm that is $\varepsilon$-close to the best one (and not exactly the best one).In this problem with several correct answers, an identification algorithm should focus on one candidate among those answers and verify that it is correct.We demonstrate that picking the answer with highest mean does not allow an algorithm to reach asymptotic optimality in terms of expected sample complexity.Instead, a \textit{furthest answer} should be identified.Using that insight to choose the candidate answer carefully, we develop a simple procedure to adapt best-arm identification algorithms to tackle $\varepsilon$-best-answer identification in transductive linear stochastic bandits. Finally, we propose an asymptotically optimal algorithm for this setting, which is shown to achieve competitive empirical performance against existing modified best-arm identification algorithms.

> Poster: On the Convergence of Local Stochastic Compositional Gradient Descent with Momentum
> 
> Authors: Hongchang Gao and Junyi Li and Heng Huang
> 
> Abstract: Federated Learning has been actively studied due to its efficiency in numerous real-world applications in the past few years. However, the federated stochastic compositional optimization problem is still an understudied problem, even though it has widespread applications in machine learning. In this paper, we developed a novel local stochastic compositional gradient descent with momentum method, which facilitates Federated Learning for the stochastic compositional problem. Importantly, we investigated the convergence rate of our proposed method and proved that  it can achieve the $O(1/\epsilon^4)$ sample complexity, which is better than  existing methods. Meanwhile, our communication complexity $O(1/\epsilon^3)$  can match   existing methods. To the best of our knowledge, this is the first work achieving such favorable sample and communication complexities. Additionally, the extensive experimental results further demonstrate the superior empirical performance over existing methods, confirming the efficacy of our methods.

> Poster: Human-in-the-loop: Provably Efficient Preference-based Reinforcement Learning with General Function Approximation
> 
> Authors: Xiaoyu Chen and Han Zhong and Zhuoran Yang and Zhaoran Wang and Liwei Wang
> 
> Abstract: We study human-in-the-loop reinforcement learning (RL) with trajectory preferences, where instead of receiving a numeric reward at each step, the RL agent only receives preferences over trajectory pairs from a human overseer. The goal of the RL agent is to learn the optimal policy which is most preferred by the human overseer. Despite the empirical success in various real-world applications, the theoretical understanding of preference-based RL (PbRL) is only limited to the tabular case.  In this paper, we propose the first optimistic model-based algorithm for PbRL with general function approximation, which estimates the model using value-targeted regression and calculates the exploratory policies by solving an optimistic planning problem. We prove that our algorithm achieves the regret bound of $\tilde{O} (\operatorname{poly}(d H) \sqrt{K} )$, where $d$ is the complexity measure of the transition and preference model depending on the Eluder dimension and log-covering numbers, $H$ is the planning horizon,  $K$ is the number of episodes, and $\tilde O(\cdot)$ omits logarithmic terms. Our lower bound indicates that our algorithm is near-optimal when specialized to the linear setting. Furthermore, we extend the PbRL problem by formulating a novel problem called RL with $n$-wise comparisons, and provide the first sample-efficient algorithm for this new setting. To the best of our knowledge, this is the first theoretical result for PbRL with (general) function approximation.

> Poster: A State-Distribution Matching Approach to Non-Episodic Reinforcement Learning
> 
> Authors: Archit Sharma and Rehaan Ahmad and Chelsea Finn
> 
> Abstract: While reinforcement learning (RL) provides a framework for learning through trial and error, translating the algorithms into the real-world has remained challenging. A major hurdle to real-world application arises from the development of algorithms in an episodic setting, in contrast to the non-episodic nature of the real-world encountered by embodied agents such as humans and robots. In this work, we focus on the autonomous RL (ARL) problem setting, where the agent is tasked with learning the task-policy in a non-episodic setting without extrinsic interventions to reset the environment after every trial. Enabling the agent to repeatedly practice the task first requires consideration of the state distribution that the agent should practice from, and how to construct it. We consider an alternating approach such that the forward policy learns to solve the task and the backward policy creates an initial state distribution to practice the task from, but what initial state distribution should the backward policy target? Assuming access to a few demonstrations, we propose a new method, MEDAL, that trains the backward policy to match the state distribution in the provided demonstrations. This keeps the agent close to the task-relevant states, allowing for a mix of easy and difficult starting states for the forward policy. Our empirical results show that MEDAL outperforms prior work on the EARL benchmark, with 40\% gains on the hardest task while making fewer assumptions than prior works, paving the way for simple and effective ARL algorithms.


> Poster: Learning to Separate Voices by Spatial Regions
> 
> Authors: Zhongweiyang Xu and Romit Roy Choudhury
> 
> Abstract: We consider the problem of audio source separation for binaural applications, such as earphones and hearing aids. While today's neural networks perform remarkably well (separating 5+ sources with 2 microphones) they assume a known or fixed maximum number of sources, $K$. Moreover, today's models are trained in a supervised manner, using training data synthesized from generic sources, environments, and human head shapes. This paper intends to relax both these constraints at the expense of a slight alteration in the problem definition. We observe that, when there are a large number of sources, it is still helpful to separate the signals by region, i.e., isolating a signal mixture for each conical sector around the user's head. This implies that we need to learn the fine-grained spatial properties of each region, including the signal distortions imposed by a person's face. We propose a two-stage self-supervised framework in which overheard signals from earphones are first selectively separated into sounds spatially, which in turn help to learn the personalized, fine-grained spatial properties of each region. Results show promising performance, while relaxing the assumptions on $K$. We believe this result aids real-world applications in selective hearing, noise cancellation, and acoustic augmented reality.

> Poster: Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context for Speech Recognition and Understanding
> 
> Authors: Yifan Peng and Siddharth Dalmia and Ian Lane and Shinji Watanabe
> 
> Abstract: Conformer has proven to be effective in many speech processing tasks. It combines the benefits of extracting local dependencies using convolutions and global dependencies using self-attention. Inspired by this, we propose a more flexible, interpretable and customizable encoder alternative, Branchformer, with parallel branches for modeling various ranged dependencies in end-to-end speech processing. In each encoder layer, one branch employs self-attention or its variant to capture long-range dependencies, while the other branch utilizes an MLP module with convolutional gating (cgMLP) to extract local relationships. We conduct experiments on several speech recognition and spoken language understanding benchmarks. Results show that our model outperforms both Transformer and vanilla cgMLP. It also matches or outperforms state-of-the-art results achieved by Conformer. Furthermore, we show various strategies to reduce computation thanks to the two-branch architecture, including the ability to have variable inference time complexity in a single trained model. The weights learned for merging branches indicate how local and global dependencies are utilized in different layers, which benefits model designing.


> Poster: Estimation in Rotationally Invariant Generalized Linear Models via Approximate Message Passing
> 
> Authors:   and Kevin Kögler and Marco Mondelli
> 
> Abstract: We consider the problem of signal estimation in generalized linear models defined via rotationally invariant design matrices. Since these matrices can have an arbitrary spectral distribution, this model is well suited to capture complex correlation structures which often arise in applications. We propose a novel family of approximate message passing (AMP) algorithms for signal estimation, and rigorously characterize their performance in the high-dimensional limit via a state evolution recursion. Assuming knowledge of the design matrix spectrum, our rotationally invariant AMP has complexity of the same order as the existing AMP derived under the restrictive assumption of a Gaussian design; our algorithm also recovers this existing AMP as a special case. Numerical results showcase a performance close to Vector AMP (which is conjectured to be Bayes-optimal in some settings), but obtained with a much lower complexity, as the proposed algorithm does not require a computationally expensive singular value decomposition.


> Poster: The Infinite Contextual Graph Markov Model
> 
> Authors: Daniele Castellana and Federico Errica and Davide Bacciu and Alessio Micheli
> 
> Abstract: The Contextual Graph Markov Model (CGMM) is a deep, unsupervised, and probabilistic model for graphs that is trained incrementally on a layer-by-layer basis. As with most Deep Graph Networks, an inherent limitation is the need to perform an extensive model selection to choose a proper size of each layer's latent representation. In this paper, we address this problem by introducing the Infinite Contextual Graph Markov Model (iCGMM), the first deep Bayesian nonparametric model for graph learning. During training, iCGMM can adapt the complexity of each layer to better fit the underlying data distribution. On 8 graph classification tasks, we show that iCGMM: i) successfully recovers or improves CGMM's performances while reducing the hyper-parameters' search space; ii) performs comparably to most end-to-end supervised methods. The results include studies on the importance of depth, hyper-parameters, and compression of the graph embeddings. We also introduce a novel approximated inference procedure that better deals with larger graph topologies.


> Poster: Knowledge-Grounded Self-Rationalization via Extractive and Natural Language Explanations
> 
> Authors: Bodhisattwa Prasad Majumder and Oana-Maria Camburu and Thomas Lukasiewicz and Julian McAuley
> 
> Abstract: An increasing number of works focus on building models that generate extractive rationales (i.e., subsets of features) or natural language explanations (NLEs) for their predictions. While an extractive rationale provides a quick view of the features most responsible for a prediction, an NLE allows for a comprehensive description of the decision-making process behind a prediction. However, current models that generate the best extractive rationales or NLEs often fall behind the state-of-the-art (SOTA) in terms of task performance. In this work, we bridge this gap by introducing RExC, a self-rationalizing framework that grounds its predictions and two complementary types of explanations (NLEs and extractive rationales) in background knowledge. Our framework improves over previous methods by: (i) reaching SOTA task performance while also providing explanations, (ii) providing two types of explanations while existing models usually provide only one type, and (iii) beating by a large margin the previous SOTA in terms of quality of explanations. Furthermore, a perturbation analysis in RExC shows a high degree of association between explanations and predictions, a necessary property of faithful explanations.


> Poster: A Completely Tuning-Free and Robust Approach to Sparse Precision Matrix Estimation
> 
> Authors: Chau Tran and Guo Yu
> 
> Abstract: Despite the vast literature on sparse Gaussian graphical models, current methods either is asymptotically tuning-free (which still requires fine-tuning in practice) or hinges on computationally expensive methods (e.g., cross-validation) to determine the proper level of regularization. We propose a completely tuning-free approach to estimating sparse Gaussian graphical models. Our method uses model-agnostic regularization parameters to estimate each column of the target precision matrix and enjoys several desirable properties. Computationally, our estimator can be computed efficiently by linear programming. Theoretically, the proposed estimator is minimax optimal under various norms. We further propose a second-stage enhancement with non-convex penalties, which possesses strong oracle properties. Through comprehensive numerical studies, our methods demonstrate favorable statistical performance. Remarkably, our methods exhibit strong robustness to the violation of the Gaussian assumption and significantly outperform competing methods in heavy-tailed settings.


> Poster: ActiveHedge: Hedge meets Active Learning
> 
> Authors: Bhuvesh Kumar and Jacob Abernethy and Venkatesh Saligrama
> 
> Abstract: We consider the classical problem of multi-class prediction with expert advice, but with an active learning twist. In this new setting the learner will only query the labels of a small number of examples, but still aims to minimize regret to the best expert as usual; the learner is also allowed a very short ``burn-in'' phase where it can fast-forward and query certain highly-informative examples. We design an algorithm that utilizes Hedge (aka Exponential Weights) as a subroutine, and we show that under a very particular combinatorial constraint on the matrix of expert predictions we can obtain a very strong regret guarantee while querying very few labels. This constraint, which we refer to as $\zeta$-compactness, or just compactness, can be viewed as a non-stochastic variant of the disagreement coefficient, another popular parameter used to reason about the sample complexity of active learning in the IID setting. We also give a polynomial time algorithm to calculate the $\zeta$-compactness of a matrix up to an approximation factor of 3.

> Poster: Generalization and Robustness Implications in Object-Centric Learning
> 
> Authors: Andrea Dittadi and Samuele Papa and Michele De Vita and Bernhard Schölkopf and Ole Winther and Francesco Locatello
> 
> Abstract: The idea behind object-centric representation learning is that natural scenes can better be modeled as compositions of objects and their relations as opposed to distributed representations. This inductive bias can be injected into neural networks to potentially improve systematic generalization and learning efficiency of downstream tasks in scenes with multiple objects. In this paper, we train state-of-the-art unsupervised models on five common multi-object datasets and evaluate segmentation accuracy and downstream object property prediction. In addition, we study systematic generalization and robustness by investigating the settings where either single objects are out-of-distribution---e.g., having unseen colors, textures, and shapes---or global properties of the scene are altered---e.g., by occlusions, cropping, or increasing the number of objects. From our experimental study, we find object-centric representations to be generally useful for downstream tasks and robust to shifts in the data distribution, especially if shifts affect single objects.


> Poster: NeuralEF: Deconstructing Kernels by Deep Neural Networks
> 
> Authors: Zhijie Deng and Jiaxin Shi and Jun Zhu
> 
> Abstract: Learning the principal eigenfunctions of an integral operator defined by a kernel and a data distribution is at the core of many machine learning problems. Traditional nonparametric solutions based on the Nystr{\"o}m formula suffer from scalability issues. Recent work has resorted to a parametric approach, i.e., training neural networks to approximate the eigenfunctions. However, the existing method relies on an expensive orthogonalization step and is difficult to implement. We show that these problems can be fixed by using a new series of objective functions that generalizes the EigenGame to function space. We test our method on a variety of supervised and unsupervised learning problems and show it provides accurate approximations to the eigenfunctions of polynomial, radial basis, neural network Gaussian process, and neural tangent kernels. Finally, we demonstrate our method can scale up linearised Laplace approximation of deep neural networks to modern image classification datasets through approximating the Gauss-Newton matrix. 


> Poster: Generic Coreset for Scalable Learning of Monotonic Kernels: Logistic Regression, Sigmoid and more
> 
> Authors: Elad Tolochinksy and Ibrahim Jubran and Dan Feldman
> 
> Abstract: Coreset (or core-set) is a small weighted \emph{subset} $Q$ of an input set $P$ with respect to a given \emph{monotonic} function $f:\mathbb{R}\to\mathbb{R}$ that \emph{provably} approximates its fitting loss $\sum_{p\in P}f(p\cdot x)$ to \emph{any} given $x\in\mathbb{R}^d$. Using $Q$ we can obtain an approximation of $x^*$ that minimizes this loss, by running \emph{existing} optimization algorithms on $Q$. In this work we provide: (i) A lower bound which proves that there are sets with no coresets smaller than $n=|P|$ for general monotonic loss functions. (ii) A proof that, under a natural assumption that holds e.g. for logistic regression and the sigmoid activation functions, a small coreset exists for \emph{any} input $P$. (iii) A generic coreset construction algorithm that computes such a small coreset $Q$ in $O(nd+n\log n)$ time, and (iv) Experimental results with open-source code which demonstrate that our coresets are effective and are much smaller in practice than predicted in theory.

> Poster: Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging
> 
> Authors: Anastasios Angelopoulos and Amit Pal Kohli and Stephen Bates and Michael Jordan and Jitendra Malik and Thayer Alshaabi and Srigokul Upadhyayula and Yaniv Romano
> 
> Abstract: Image-to-image regression is an important learning task, used frequently in biological imaging. Current algorithms, however, do not generally offer statistical guarantees that protect against a model's mistakes and hallucinations. To address this, we develop uncertainty quantification techniques with rigorous statistical guarantees for image-to-image regression problems. In particular, we show how to derive uncertainty intervals around each pixel that are guaranteed to contain the true value with a user-specified confidence probability. Our methods work in conjunction with any base machine learning model, such as a neural network, and endow it with formal mathematical guarantees—regardless of the true unknown data distribution or choice of model. Furthermore, they are simple to implement and computationally inexpensive. We evaluate our procedure on three image-to-image regression tasks: quantitative phase microscopy, accelerated magnetic resonance imaging, and super-resolution transmission electron microscopy of a Drosophila melanogaster brain.


> Poster: Bitwidth Heterogeneous Federated Learning with Progressive Weight Dequantization
> 
> Authors: Jaehong Yoon and Geon Park and Wonyong Jeong and Sung Ju Hwang
> 
> Abstract: In practical federated learning scenarios, the participating devices may have different bitwidths for computation and memory storage by design. However, despite the progress made in device-heterogeneous federated learning scenarios, the heterogeneity in the bitwidth specifications in the hardware has been mostly overlooked. We introduce a pragmatic FL scenario with bitwidth heterogeneity across the participating devices, dubbed as Bitwidth Heterogeneous Federated Learning (BHFL). BHFL brings in a new challenge, that the aggregation of model parameters with different bitwidths could result in severe performance degeneration, especially for high-bitwidth models. To tackle this problem, we propose ProWD framework, which has a trainable weight dequantizer at the central server that progressively reconstructs the low-bitwidth weights into higher bitwidth weights, and finally into full-precision weights. ProWD further selectively aggregates the model parameters to maximize the compatibility across bit-heterogeneous weights. We validate ProWD against relevant FL baselines on the benchmark datasets, using clients with varying bitwidths. Our ProWD largely outperforms the baseline FL algorithms as well as naive approaches (e.g. grouped averaging) under the proposed BHFL scenario.


> Poster: On the Finite-Time Performance of the Knowledge Gradient Algorithm
> 
> Authors: Yanwen Li and Siyang Gao
> 
> Abstract: The knowledge gradient (KG) algorithm is a popular and effective algorithm for the best arm identification (BAI) problem. Due to the complex calculations of KG, theoretical analysis of this algorithm is difficult, and existing results are mostly about the asymptotic performance of it, e.g., consistency, asymptotic sample allocation, etc. In this research, we present new theoretical results about the finite-time performance of the KG algorithm. Under independent and normally distributed rewards, we derive lower bounds and upper bounds for the probability of error and simple regret of the algorithm. With these bounds, existing asymptotic results become simple corollaries. We also show the performance of the algorithm for the multi-armed bandit (MAB) problem. These developments not only extend the existing analysis of the KG algorithm, but can also be used to analyze other improvement-based algorithms. Last, we use numerical experiments to further demonstrate the finite-time behavior of the KG algorithm.


> Poster: QSFL: A Two-Level Uplink Communication Optimization Framework for Federated Learning
> 
> Authors: Liping Yi and Wang Gang and Liu Xiaoguang
> 
> Abstract: In cross-device Federated Learning (FL), the billing and unreliable low-bandwidth connections between edge devices and the server make the communication cost caused by transmitting full-precision local models be a significant bottleneck. We propose a novel framework termed QSFL, aims to optimize FL uplink (client-to-server) communication at both client and model levels. At the client level, we design Qualification Judgment (QJ) to sample high-qualification clients for uploading models. At the model level, we explore Sparse Cyclic Sliding Segment (SCSS) to further compress transmitted models. We prove QSFL can converge over wall-to-wall time, and develop an optimal hyperparameter searching algorithm based on theoretical analysis, towards enabling QSFL to make the best trade-off between model accuracy and communication cost. Experimental results show that QSFL achieves the state-of-the-art compression ratio with marginal model accuracy degradation.


> Poster: Expression might be enough: representing pressure and demand for reinforcement learning based traffic signal control
> 
> Authors: Liang Zhang and Qiang Wu and Jun Shen and Linyuan Lü and Bo Du and Jianqing Wu
> 
> Abstract: Many studies confirmed that a proper traffic state representation is more important than complex algorithms for the classical traffic signal control (TSC) problem. In this paper, we (1) present a novel, flexible and efficient method, namely advanced max pressure (Advanced-MP), taking both running and queuing vehicles into consideration to decide whether to change current signal phase; (2) inventively design the traffic movement representation with the efficient pressure and effective running vehicles from Advanced-MP, namely advanced traffic state (ATS); and (3) develop a reinforcement learning (RL) based algorithm template, called Advanced-XLight, by combining ATS with the latest RL approaches, and generate two RL algorithms, namely "Advanced-MPLight" and "Advanced-CoLight" from Advanced-XLight. Comprehensive experiments on multiple real-world datasets show that:  (1) the Advanced-MP outperforms baseline methods, and it is also efficient and reliable for deployment; and (2) Advanced-MPLight and Advanced-CoLight can achieve the state-of-the-art.


> Poster: Transfer Learning In Differential Privacy's Hybrid-Model
> 
> Authors: Or Sheffet and Refael Kohen
> 
> Abstract: The \emph{hybrid-model} (Avent et al 2017) in Differential Privacy is a an augmentation of the local-model where in addition to $N$ local-agents we are assisted by one special agent who is in fact a curator holding the sensitive details of $n$ additional individuals. Here we study the problem of machine learning in the hybrid-model where the $n$ individuals in the curator's dataset are drawn from a \emph{different} distribution than the one of the general population (the local-agents). We give a general scheme -- Subsample-Test-Reweigh -- for this \emph{transfer learning} problem, which reduces any curator-model learner to a learner in the hybrid-model using iterative subsampling and reweighing of the $n$ examples held by the curator based on a smooth variation (introduced by Bun et al 2020) of the Multiplicative-Weights algorithm. Our scheme has a sample complexity which relies on the $\chi^2$-divergence between the two distributions. We give worst-case analysis bounds on the sample complexity required for our private reduction. Aiming to reduce said sample complexity, we give two specific instances our sample complexity can be drastically reduced (one instance is analyzed mathematically, while the other - empirically) and pose several directions for follow-up work.

> Poster: Burst-dependent plasticity and dendritic amplification support target-based learning and hierarchical imitation learning
> 
> Authors: Cristiano Capone and Cosimo Lupo and Paolo Muratore and Pier Stanislao Paolucci
> 
> Abstract: The brain can learn a wide range of tasks very efficiently in terms of energy consumption, motivating the search for biologically inspired learning rules for improving the efficiency of artificial intelligence.Most biological models are composed of point neurons and cannot achieve the state-of-art performances of artificial intelligence (e.g. they struggle to solve the credit assignment problem).Recent works have proposed that segregation of dendritic input (neurons receive sensory information and higher-order feedback in segregated compartments) and generation of high-frequency bursts of spikes would support backpropagation in biological neurons.However, these approaches require propagating errors with a fine spatio-temporal structure to all the neurons. It is not clear whether this is possible in biological networks. For this reason, in the last few years, target-based approaches started to gain more and more interest.We propose that bursts and dendritic input segregation give the possibility to implement a target-based learning framework in a biologically plausible way. Indeed, target-based approaches require evaluating at the same time the spontaneous activity and the target activity of the network. In our model, this is allowed by combining dendritic segregation (only part of the dendrites receive the teaching signal) and the coincidence mechanism between basal and apical inputs generating the burst.We suggest that this neuronal architecture naturally allows for orchestrating “hierarchical imitation learning”, enabling the decomposition of challenging long-horizon decision-making tasks into simpler subtasks.We demonstrated this in the button-and-food task (first reach the button, then the food) by separating the high-level network that elaborates the strategy selecting the subtask (whether to reach the button or the food) and produces contextual and abstract signals to coordinate the low-level network which actuates the execution (the reach subtask).


> Poster: Communication-Efficient Adaptive Federated Learning
> 
> Authors: Yujia Wang and Lu Lin and Jinghui Chen
> 
> Abstract: Federated learning is a machine learning training paradigm that enables clients to jointly train models without sharing their own localized data. However, the implementation of federated learning in practice still faces numerous challenges, such as the large communication overhead due to the repetitive server-client synchronization and the lack of adaptivity by SGD-based model updates. Despite that various methods have been proposed for reducing the communication cost by gradient compression or quantization, and the federated versions of adaptive optimizers such as FedAdam are proposed to add more adaptivity, the current federated learning framework still cannot solve the aforementioned challenges all at once. In this paper, we propose a novel communication-efficient adaptive federated learning method (FedCAMS) with theoretical convergence guarantees. We show that in the nonconvex stochastic optimization setting, our proposed FedCAMS achieves the same convergence rate of $\cO(\frac{1}{\sqrt{TKm}})$ as its non-compressed counterparts. Extensive experiments on various benchmarks verify our theoretical analysis.

> Poster: Federated Learning with Label Distribution Skew via Logits Calibration
> 
> Authors: Jie Zhang and Zhiqi Li and Bo Li and Jianghe Xu and Shuang Wu and Shouhong Ding and Chao Wu
> 
> Abstract: Traditional federated optimization methods perform poorly with heterogeneous data (i.e.\ , accuracy reduction), especially for highly skewed data. In this paper, we investigate the label distribution skew in FL, where the distribution of labels varies across clients. First, we investigate the label distribution skew from a statistical view. We demonstrate both theoretically and empirically that previous methods based on softmax cross-entropy are not suitable, which can result in local models heavily overfitting to minority classes and missing classes. Additionally, we theoretically introduce a deviation bound to measure the deviation of the gradient after local update. At last, we propose FedLC (\textbf{Fed}erated learning via \textbf{L}ogits \textbf{C}alibration), which calibrates the logits before softmax cross-entropy according to the probability of occurrence of each class. FedLC applies a fine-grained calibrated cross-entropy loss to local update by adding a pairwise label margin. Extensive experiments on federated datasets and real-world datasets demonstrate that FedLC leads to a more accurate global model and much improved performance. Furthermore, integrating other FL methods into our approach can further enhance the performance of the global model.


> Poster: PMIC: Improving Multi-Agent Reinforcement Learning with ProgressiveMutual Information Collaboration
> 
> Authors: Pengyi Li and Hongyao Tang and Tianpei Yang and Xiaotian Hao and Tong Sang and Yan Zheng and Jianye Hao and Matthew Taylor and Wenyuan Tao and Zhen Wang
> 
> Abstract: Learning to collaborate is critical in multi-agent reinforcement learning (MARL). A number of previous works promote collaboration by maximizing the correlation of agents’ behaviors, which is typically characterised by mutual information (MI) in different forms. However, in this paper, we reveal that strong correlation can emerge from sub-optimal collaborative behaviors, and simply maximizing the MI can, surprisingly, hinder the learning towards better collaboration. To address this issue, we propose a novel MARL framework, called Progressive Mutual Information Collaboration (PMIC), for more effective MI-driven collaboration. In PMIC, we use a new collaboration criterion measured by the MI between global states and joint actions. Based on the criterion, the key idea of PMIC is maximizing the MI associated with superior collaborative behaviors and minimizing the MI associated with inferior ones. The two MI objectives play complementary roles by facilitating learning towards better collaborations while avoiding falling into sub-optimal ones. Specifically, PMIC stores and progressively maintains sets of superior and inferior interaction experiences, from which dual MI neural estimators are established. Experiments on a wide range of MARL benchmarks show the superior performance of PMIC compared with other algorithms.


> Poster: Forward Operator Estimation in Generative Models with Kernel Transfer Operators
> 
> Authors: Zhichun Huang and Rudrasis Chakraborty and Vikas Singh
> 
> Abstract: Generative models which use explicit density modeling (e.g., variational autoencoders, flow-based generative models) involve finding a mapping from a known distribution, e.g. Gaussian, to the unknown input distribution. This often requires searching over a class of non-linear functions (e.g., representable by a deep neural network). While effective in practice, the associated runtime/memory costs can increase rapidly, usually as a function of the performance desired in an application. We propose a substantially cheaper (and simpler) forward operator estimation strategy based on adapting known results on kernel transfer operators. We show that our formulation enables highly efficient distribution approximation and sampling, and offers surprisingly good empirical performance that compares favorably with powerful baselines, but with significant runtime savings. We show that the algorithm also performs well in small sample size settings (in brain imaging). 


> Poster: Extracting Latent State Representations with Linear Dynamics from Rich Observations
> 
> Authors: Abraham Frandsen and Rong Ge and Holden Lee
> 
> Abstract: Recently, many reinforcement learning techniques have been shown to have provable guarantees in the simple case of linear dynamics, especially in problems like linear quadratic regulators. However, in practice many tasks require learning a policy from rich, high-dimensional features such as images, which are unlikely to be linear. We consider a setting where there is a hidden linear subspace of the high-dimensional feature space in which the dynamics are linear. We design natural objectives based on forward and inverse dynamics models. We prove that these objectives can be efficiently optimized and their local optimizers extract the hidden linear subspace. We empirically verify our theoretical results with synthetic data and explore the effectiveness of our approach (generalized to nonlinear settings) in simple control tasks with rich observations.


> Poster: Neural language models are not born equal to fit brain data, but training helps
> 
> Authors: Alexandre Pasquiou and Yair Lakretz and Christophe Pallier and Thirion Bertrand and John Hale
> 
> Abstract: Neural Language Models (NLMs) have made tremendous advances during the last years, achieving impressive performance on various linguistic tasks.Capitalizing on this, studies in neuroscience have started to use NLMs to study neural activity in the human brain during language processing.However, many questions remain unanswered regarding which factors determine the ability of a neural language model to capture brain activity (aka, its 'brain score').Here, we make first steps in this direction and examine the impact of test loss, training corpus and model architecture (comparing GloVe, LSTM, GPT-2 and BERT), on the prediction of functional Magnetic Resonance Imaging timecourses of participants listening to an audiobook.We find that (1) untrained versions of each model already explain significant amount of signal in the brain, with the untrained LSTM outperforming the others; (2) that training NLP models improves brain scores in the same brain regions irrespective of the model's architecture; (3) that Perplexity (test loss) is not a good predictor of brain score; (4) that training data have a strong influence on the outcome and, notably, that off-the-shelves models may lack statistical power to detect brain activations. Overall, we outline the impact of model-training choices, and suggest good practices for future studies aiming at explaining the human language system using neural language models.


> Poster: UAST: Uncertainty-Aware Siamese Tracking
> 
> Authors: Dawei Zhang and Yanwei Fu and Zhonglong Zheng
> 
> Abstract: Visual object tracking is basically formulated as target classification and bounding box estimation. Recent anchor-free Siamese trackers rely on predicting the distances to four sides for efficient regression but fail to estimate accurate bounding box in complex scenes. We argue that these approaches lack a clear probabilistic explanation, so it is desirable to model the uncertainty and ambiguity representation of target estimation. To address this issue, this paper presents an Uncertainty-Aware Siamese Tracker (UAST) by developing a novel distribution based regression formulation with localization uncertainty. We exploit regression vectors to directly represent the discretized probability distribution for four offsets of boxes, which is general, flexible and informative. Based on the resulting distributed representation, our method is able to provide a probabilistic value of uncertainty. Furthermore, considering the high correlation between the uncertainty and regression accuracy, we propose to learn a joint representation head of classification and localization quality for reliable tracking, which also avoids the inconsistency of classification and quality estimation between training and inference. Extensive experiments on several challenging tracking benchmarks demonstrate the effectiveness of UAST and its superiority over other Siamese trackers. Notably, our improvements are almost cost-free.


> Poster: EqR: Equivariant Representations for Data-Efficient Reinforcement Learning
> 
> Authors: Arnab Kumar Mondal and Vineet Jain and Kaleem Siddiqi and Siamak Ravanbakhsh
> 
> Abstract: We study different notions of equivariance as an inductive bias in Reinforcement Learning (RL) and propose new mechanisms for recovering representations that are equivariant to both an agent’s action, and symmetry transformations of the state-action pairs. Whereas prior work on exploiting symmetries in deep RL can only incorporate predefined linear transformations, our approach allows for non- linear symmetry transformations of state-action pairs to be learned from the data itself. This is achieved through an equivariant Lie algebraic parameterization of state and action encodings, equivariant latent transition models, and the use of symmetry-based losses. We demonstrate the advantages of our learned equivari- ant representations for Atari games, in a data-efficient setting limited to 100K steps of interactions with the environment. Our method, which we call Equiv- ariant representations for RL (EqR), outperforms other comparable methods on statistically reliable evaluation metrics.


> Poster: Fenrir: Physics-Enhanced Regression for Initial Value Problems
> 
> Authors: Filip Tronarp and Nathanael Bosch and Philipp Hennig
> 
> Abstract: We show how probabilistic numerics can be used to convert an initial value problem into a Gauss--Markov process parametrised by the dynamics of the initial value problem. Consequently, the often difficult problem of parameter estimation in ordinary differential equations is reduced to hyper-parameter estimation in Gauss--Markov regression, which tends to be considerably easier. The method's relation and benefits in comparison to classical numerical integration and gradient matching approaches is elucidated. In particular, the method can, in contrast to gradient matching, handle partial observations, and has certain routes for escaping local optima not available to classical numerical integration. Experimental results demonstrate that the method is on par or moderately better than competing approaches.


> Poster: An Exact Symbolic Reduction of Linear Smart Predict+Optimize to Mixed Integer Linear Programming
> 
> Authors: Jihwan Jeong and Parth Jaggi and Andrew Butler and Scott Sanner
> 
> Abstract: Predictive models are traditionally optimized independently of their use in downstream decision-based optimization. The `smart, predict then optimize' (SPO) framework addresses this shortcoming by optimizing predictive models in order to minimize the final downstream decision loss. To date, several local first-order methods and convex approximations have been proposed. These methods have proven to be effective in practice, however, it remains generally unclear as to how close these local solutions are to global optimality. In this paper, we cast the SPO problem as a bi-level program and apply Symbolic Variable Elimination (SVE) to analytically solve the lower optimization.  The resulting program can then be formulated as a MILP which is solved to global optimality using standard off-the-shelf solvers. To our knowledge, our framework is the first to provide a globally optimal solution to the linear SPO problem.  Experimental results comparing with state-of-the-art local SPO solvers show that the globally optimal solution obtains up to two orders of magnitude reduction in decision regret.


> Poster: Label-Free Explainability for Unsupervised Models
> 
> Authors: Jonathan Crabbé and Mihaela van der Schaar
> 
> Abstract: Unsupervised black-box models are challenging to interpret. Indeed, most existing explainability methods require labels to select which component(s) of the black-box's output to interpret. In the absence of labels, black-box outputs often are representation vectors whose components do not correspond to any meaningful quantity. Hence, choosing which component(s) to interpret in a label-free unsupervised/self-supervised setting is an important, yet unsolved problem. To bridge this gap in the literature, we introduce two crucial extensions of post-hoc explanation techniques: (1) label-free feature importance and (2) label-free example importance that respectively highlight influential features and training examples for a black-box to construct representations at inference time. We demonstrate that our extensions can be successfully implemented as simple wrappers around many existing feature and example importance methods. We illustrate the utility of our label-free explainability paradigm through a qualitative and quantitative comparison of representation spaces learned by various autoencoders trained on distinct unsupervised tasks. 


> Poster: When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee
> 
> Authors: Dixian Zhu and Gang Li and Bokun Wang and Xiaodong Wu and Tianbao Yang
> 
> Abstract: In this paper, we propose systematic and efficient gradient-based methods for both one-way and two-way partial AUC (pAUC) maximization that are applicable to deep learning. We propose new formulations of pAUC surrogate objectives by using the distributionally robust optimization (DRO) to define the loss for each individual positive data. We consider two formulations of DRO, one of which is based on conditional-value-at-risk (CVaR) that yields a non-smooth but exact estimator for pAUC, and another one is based on a KL divergence regularized DRO that yields an inexact but smooth (soft) estimator for pAUC. For both one-way and two-way pAUC maximization, we propose two algorithms and prove their convergence for optimizing their two formulations, respectively. Experiments demonstrate the effectiveness of the proposed algorithms for pAUC maximization for deep learning on various datasets. 


> Poster: Correlated quantization for distributed mean estimation and optimization
> 
> Authors: Ananda Suresh and Ziteng Sun and Jae Ro and Felix Xinnan Yu
> 
> Abstract: We study the problem of distributed mean estimation and optimization under communication constraints. We propose a correlated quantization protocol whose error guarantee depends on the deviation of data points instead of their absolute range. The design doesn't need any prior knowledge on the concentration property of the dataset, which is required to get such dependence in previous works.  We show that applying the proposed protocol as a sub-routine in distributed optimization algorithms leads to better convergence rates.  We also prove the optimality of our protocol under mild assumptions. Experimental results show that our proposed algorithm outperforms existing mean estimation protocols on a diverse set of tasks.


> Poster: Robust Multi-Objective Bayesian Optimization Under Input Noise
> 
> Authors: Samuel Daulton and Sait Cakmak and Maximilian Balandat and Michael A Osborne and Enlu Zhou and Eytan Bakshy
> 
> Abstract: Bayesian optimization (BO) is a sample-efficient approach for tuning design parameters to optimize expensive-to-evaluate, black-box performance metrics. In many manufacturing processes, the design parameters are subject to random input noise, resulting in a product that is often less performant than expected. Although BO methods have been proposed for optimizing a single objective under input noise, no existing method addresses the practical scenario where there are multiple objectives that are sensitive to input perturbations. In this work, we propose the first multi-objective BO method that is robust to input noise. We formalize our goal as optimizing the multivariate value-at-risk (MVaR), a risk measure of the uncertain objectives. Since directly optimizing MVaR is computationally infeasible in many settings, we propose a scalable, theoretically-grounded approach for optimizing MVaR using random scalarizations. Empirically, we find that our approach significantly outperforms alternative methods and efficiently identifies optimal robust designs that will satisfy specifications across multiple metrics with high probability.


> Poster: Adaptive Gaussian Process Change Point Detection
> 
> Authors: Edoardo Caldarelli and Philippe Wenk and Stefan Bauer and Andreas Krause
> 
> Abstract: Detecting change points in time series, i.e., points in time at which some observed process suddenly changes, is a fundamental task that arises in many real-world applications, with consequences for safety and reliability. In this work, we propose $\operatorname{ADAGA}$, a novel Gaussian process-based solution to this problem, that leverages a powerful heuristics we developed based on statistical hypothesis testing. In contrast to prior approaches, $\operatorname{ADAGA}$ adapts to changes both in mean and covariance structure of the temporal process. In extensive experiments, we show its versatility and applicability to different classes of change points, demonstrating that it is significantly more accurate than current state-of-the-art alternatives.

> Poster: Removing Batch Normalization Boosts Adversarial Training
> 
> Authors: Haotao Wang and Aston Zhang and Shuai Zheng and Xingjian Shi and Mu Li and Zhangyang Wang
> 
> Abstract: A critical challenge of adversarial training (AT) is the ``mixture distribution challenge'', which refers to training a model simultaneously over clean and adversarial images that are from two different underlying distributions. Previous works show traditional batch normalization (BN) to be a major bottleneck, since it is challenging to estimate normalization statistics of such mixture distributions in AT. The previous state-of-the-art solution, termed multiple BN (MBN), uses two separate sets of BNs for clean and adversarial images respectively. Despite being effective, MBN has a major limitation: Different test samples require different BN sets to achieve the best prediction, while there is no oracle to indicate which BN set to use for each test sample during inference. Besides, the optimal number of BN sets to use in MBN is unknown: Adding more sets of BN in MBN does not always add value. To address these limitations, we proposed normalization-free adversarial training (NFAT), which explores a brand-new avenue to remove all BNs in AT. Experimental results demonstrate that NFAT significantly outperforms previous state-of-the-art AT methods. For example, NFAT achieves $72.67\%$ clean accuracy on ImageNet using ResNet50, at comparable or better robustness against multiple adversarial attacks than previous state-of-the-art AT methods based on BN, which have at most $60.52\%$ clean accuracy. Source code and pre-trained models will be released. 

> Poster: Guarantees for Epsilon-Greedy Reinforcement Learning with Function Approximation
> 
> Authors: Chris Dann and Yishay Mansour and Mehryar Mohri and Ayush Sekhari and Karthik Sridharan
> 
> Abstract: Myopic exploration policies such as epsilon-greedy, softmax, or Gaussian noise fail to explore efficiently in some reinforcement learning tasks and yet, they perform well in many others. In fact, in practice, they are often selected as the top choices, due to their simplicity. But, for what tasks do such policies succeed? Can we give theoretical guarantees for their favorable performance? These crucial questions have been scarcely investigated, despite the prominent practical importance of these policies. This paper presents a theoretical analysis of such policies and provides the first regret and sample-complexity bounds for reinforcement learning with myopic exploration. Our results apply to value-function-based algorithms in episodic MDPs with bounded Bellman Eluder dimension.  We propose a new complexity measure  called myopic exploration gap, denoted by \alpha, that captures a structural property of the MDP, the exploration policy and the given value function class. We show that the sample-complexity of myopic exploration scales quadratically with the inverse of this quantity, 1 / \alpha^2. We further demonstrate through concrete examples that myopic exploration gap is indeed favorable in several tasks where myopic exploration succeeds, due to the corresponding dynamics and reward structure.


> Poster: Deduplicating Training Data Mitigates Privacy Risks
> 
> Authors: Nikhil Kandpal and Eric Wallace and Colin Raffel
> 
> Abstract: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.


> Poster: Learning to Predict Graphs with Fused Gromov-Wasserstein Barycenters
> 
> Authors: Luc Brogat-Motte and Rémi Flamary and Celine Brouard and Juho Rousu and Florence d'Alché-Buc
> 
> Abstract: We formulate the problem as regression with the Fused Gromov-Wasserstein (FGW) loss and propose a -predictive model relying on a FGW barycenter whose weights depend on inputs. First we introduce a non-parametric estimator based on kernel ridge regression where the barycenter is computed over the training output data for which theoretical results such as consistency and excess risk bound are proved. Next we propose an interpretable parametric model where the barycenter weights are modeled with a neural network and the graphs on which the FGW barycenter is calculated are additionally learned. Numerical experiments show the strength of the method and its ability to interpolate in the labeled graph space on simulated data and on a difficult metabolic identification problem where it can reach very good performance with very little engineering.


> Poster: Matching Learned Causal Effects of Neural Networks with Domain Priors
> 
> Authors: Gowtham Reddy Abbavaram and SAI SRINIVAS KANCHETI and Vineeth N Balasubramanian and Amit Sharma
> 
> Abstract: A trained neural network can be interpreted as a structural causal model (SCM) that provides the effect of changing input variables on the model's output. However, if training data contains both causal and correlational relationships, a model that optimizes prediction accuracy may not necessarily learn the true causal relationships between input and output variables. On the other hand, expert users often have prior knowledge of the causal relationship between certain input variables and output from domain knowledge. Therefore, we propose a regularization method that aligns the learned causal effects of a neural network with domain priors, including both direct and total causal effects. We show that this approach can generalize to different kinds of domain priors, including monotonicity of causal effect of an input variable on output or zero causal effect of a variable on output for purposes of fairness. Our experiments on twelve benchmark datasets show its utility in regularizing a neural network model to maintain desired causal effects, without compromising on accuracy. Importantly, we also show that a model thus trained is robust and gets improved accuracy on noisy inputs.


> Poster: Model Selection in Batch Policy Optimization
> 
> Authors: Jonathan Lee and George Tucker and Ofir Nachum and Bo Dai
> 
> Abstract: We study the problem of model selection in batch policy optimization: given a fixed, partial-feedback dataset and M model classes, learn a policy with performance that is competitive with the policy derived from the best model class. We formalize the problem in the contextual bandit setting with linear model classes by identifying three sources of error that any model selection algorithm should optimally trade-off in order to be competitive: (1) approximation error, (2) statistical complexity, and (3) coverage. The first two sources are common in model selection for supervised learning, where optimally trading-off these properties is well-studied. In contrast, the third source is unique to batch policy optimization and is due to dataset shift inherent to the setting. We first show that no batch policy optimization algorithm can achieve a guarantee addressing all three simultaneously, revealing a stark contrast between difficulties in batch policy optimization and the positive results available in supervised learning. Despite this negative result, we show that relaxing any one of the three error sources enables the design of algorithms achieving near-oracle inequalities for the remaining two. We conclude with experiments demonstrating the efficacy of these algorithms.


> Poster: YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone
> 
> Authors: Edresson Casanova and Julian Weber and Christopher Shulby and Arnaldo Candido Junior and Eren Gölge and Moacir Ponti
> 
> Abstract: YourTTS brings the power of a multilingual approach to the task of zero-shot multi-speaker TTS. Our method builds upon the VITS model and adds several novel modifications for zero-shot multi-speaker and multilingual training. We achieved state-of-the-art (SOTA) results in zero-shot multi-speaker TTS and results comparable to SOTA in zero-shot voice conversion on the VCTK dataset. Additionally, our approach achieves promising results in a target language with a single-speaker dataset, opening possibilities for zero-shot multi-speaker TTS and zero-shot voice conversion systems in low-resource languages. Finally, it is possible to fine-tune the YourTTS model with less than 1 minute of speech and achieve state-of-the-art results in voice similarity and with reasonable quality. This is important to allow synthesis for speakers with a very different voice or recording characteristics from those seen during training.


> Poster: Algorithms for the Communication of Samples
> 
> Authors: Lucas Theis and Noureldin Yosri Yehia Ahmed
> 
> Abstract: The efficient communication of noisy data has applications in several areas of machine learning, such as neural compression or differential privacy, and is also known as reverse channel coding or the channel simulation problem. Here we propose two new coding schemes with practical advantages over existing approaches. First, we introduce ordered random coding (ORC) which uses a simple trick to reduce the coding cost of previous approaches. This scheme further illuminates a connection between schemes based on importance sampling and the so-called Poisson functional representation. Second, we describe a hybrid coding scheme which uses dithered quantization to more efficiently communicate samples from distributions with bounded support.


> Poster: Transformers are Meta-Reinforcement Learners
> 
> Authors: Luckeciano Melo
> 
> Abstract: The transformer architecture and variants presented a remarkable success across many machine learning tasks in recent years. This success is intrinsically related to the capability of handling long sequences and the presence of context-dependent weights from the attention mechanism. We argue that these capabilities suit the central role of a Meta-Reinforcement Learning algorithm. Indeed, a meta-RL agent needs to infer the task from a sequence of trajectories. Furthermore, it requires a fast adaptation strategy to adapt its policy for a new task - which can be achieved using the self-attention mechanism. In this work, we present TrMRL (Transformers for Meta-Reinforcement Learning), a meta-RL agent that mimics the memory reinstatement mechanism using the transformer architecture. It associates the recent past of working memories to build an episodic memory recursively through the transformer layers. We show that the self-attention computes a consensus representation that minimizes the Bayes Risk at each layer and provides meaningful features to compute the best actions. We conducted experiments in high-dimensional continuous control environments for locomotion and dexterous manipulation. Results show that TrMRL presents comparable or superior asymptotic performance, sample efficiency, and out-of-distribution generalization compared to the baselines in these environments.


> Poster: Utility Theory for Markovian Sequential Decision Making
> 
> Authors: Mehran Shakerinava and Siamak Ravanbakhsh
> 
> Abstract: The von Neumann-Morgenstern (VNM) utility theorem shows that under certain axioms of rationality, decision-making is reduced to maximizing the expectation of some utility function. We extend these axioms to increasingly structured sequential decision making settings and identify the structure of the corresponding utility functions. In particular, we show that a Markovian assumption leads to a utility in the form of a per transition reward and multiplicative factor on the future return. This motivates a generalization of Markov Decision Processes (MDPs) with this structure on the agent's returns, which we call Affine MDPs. A stronger constraint on preferences is needed to recover the commonly used cumulative sum of scalar rewards in MDPs. A yet stronger constraint simplifies the utility function for goal-seeking agents in the form of a difference in potentials. Our necessary and sufficient conditions demystify the reward hypothesis that underlies the design of rational agents in reinforcement learning by adding an axiom to the VNM rationality axioms and motivates new directions for AI research involving sequential decision making.


> Poster: Federated Learning with Positive and Unlabeled Data
> 
> Authors: Xinyang Lin and Hanting Chen and Yixing Xu and Chao Xu and Xiaolin Gui and Yiping Deng and Yunhe Wang
> 
> Abstract: We study the problem of learning from positive and unlabeled (PU) data in the federated setting, where each client only labels a little part of their dataset due to the limitation of resources and time. Different from the settings in traditional PU learning where the negative class consists of a single class, the negative samples which cannot be identified by a client in the federated setting may come from multiple classes which are unknown to the client. Therefore, existing PU learning methods can be hardly applied in this situation. To address this problem, we propose a novel framework, namely Federated learning with Positive and Unlabeled data (FedPU), to minimize the expected risk of multiple negative classes by leveraging the labeled data in other clients. We theoretically analyze the generalization bound of the proposed FedPU. Empirical experiments show that the FedPU can achieve much better performance than conventional supervised and semi-supervised federated learning methods.  


> Poster: Individual Reward Assisted Multi-Agent Reinforcement Learning
> 
> Authors: Li Wang and Yujing Hu and Yupeng Zhang and Weixun Wang and Chongjie Zhang and Yang Gao and Jianye Hao and Tangjie Lv and Changjie Fan
> 
> Abstract: In many real-world multi-agent systems, the sparsity of team rewards often makes it difficult for an algorithm to successfully learn a cooperative team policy. One common way for addressing this issue is to design individual rewards for the agent which enable it to learn skills that would be beneficial to cooperation. However, in most previous work, the agents' individual rewards are simply added to the team reward, which may largely change the target of the team and result in a sub-optimal policy. In this paper, we propose \emph{Individual Reward Assisted Team Policy Learning} (IRAT), which learns two policies for each agent from the dense individual rewards and the sparse team rewards with discrepancy constraints for updating the two policies mutually. Experiments in different scenarios of Multi-Agent Particle Environment and SISL Environments created at SISL (Stanford Intelligent Systems Laboratory) demonstrate that our method can greatly promote team policy to learn sparse team rewards without deviating from the original objective of team, even if individual rewards sometimes mislead or conflict with team rewards objective and outperforms state-of-the-art methods, such as MAPPO and other methods for utilizing individual rewards.


> Poster: Constrained Optimization with Dynamic Bound-scaling for Effective NLP Backdoor Defense
> 
> Authors: Guangyu Shen and Yingqi Liu and Guanhong Tao and Qiuling Xu and ZHUO ZHANG and Shengwei An and Shiqing Ma and Xiangyu Zhang
> 
> Abstract: We develop a novel optimization method for NLP backdoor inversion. We leverage a dynamically reducing temperature coefficient in the softmax function to provide changing loss landscapes to the optimizer such that the process gradually focuses on the ground truth trigger, which is denoted as a one-hot value in a convex hull. Our method also features a temperature rollback mechanismto step away from local optimals, exploiting the observation that local optimals can be easily determined in NLP trigger inversion (while not in general optimization). We evaluate the technique on over 1600 models (with roughly half of them having injected backdoors) on 3 prevailing NLP tasks, with 4 different backdoor attacks and 7 architectures. Our results show that the technique is ableto effectively and efficiently detect and remove backdoors, outperforming 4 baseline methods.


> Poster: Optimal Algorithms for Stochastic Multi-Level Compositional Optimization
> 
> Authors: Wei Jiang and Bokun Wang and Yibo Wang and Lijun Zhang and Tianbao Yang
> 
> Abstract: In this paper, we investigate the problem of stochastic multi-level compositional optimization, where the objective function is a composition of multiple smooth but possibly non-convex functions. Existing methods for solving this problem either suffer from sub-optimal sample complexities or need a huge batch size. To address this limitation, we propose a Stochastic Multi-level Variance Reduction method (SMVR), which achieves the optimal sample complexity of $\mathcal{O}\left(1 / \epsilon^{3}\right)$ to find an $\epsilon$-stationary point for non-convex objectives. Furthermore, when the objective function satisfies the convexity or Polyak-Łojasiewicz (PL) condition, we propose a stage-wise variant of SMVR and improve the sample complexity to $\mathcal{O}\left(1 / \epsilon^{2}\right)$ for convex functions or $\mathcal{O}\left(1 /(\mu\epsilon)\right)$ for non-convex functions satisfying the $\mu$-PL condition. The latter result implies the same complexity for $\mu$-strongly convex functions. To make use of adaptive learning rates, we also develop Adaptive SMVR, which achieves the same optimal complexities but converges faster in practice. All our complexities match the lower bounds not only in terms of $\epsilon$ but also in terms of $\mu$ (for PL or strongly convex functions), without using a large batch size in each iteration.

> Poster: Coordinated Double Machine Learning
> 
> Authors: Nitai Fingerhut and Yaniv Romano and Matteo Sesia
> 
> Abstract: Double machine learning is a statistical method for leveraging complex black-box models to construct approximately unbiased treatment effect estimates given observational data with high-dimensional covariates. The idea is to first fit on a subset of the samples two predictive models, one for the continuous outcome of interest and one for the observed treatment, and then to estimate the treatment effect using the remaining samples through a simple orthogonalized regression. While this methodology is flexible and can accommodate arbitrary predictive models, typically trained independently of one another, this paper argues that a carefully coordinated learning algorithm for deep neural networks may further reduce the estimation bias. The improved empirical performance of the proposed method is demonstrated through numerical experiments on both simulated and real data.


> Poster: Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning
> 
> Authors: Yang Zhao and Hao Zhang and Xiuyuan Hu
> 
> Abstract: How to train deep neural networks (DNNs) to generalize well is a central concern in deep learning, especially for severely overparameterized networks nowadays. In this paper, we propose an effective method to improve the model generalization by additionally penalizing the gradient norm of loss function during optimization. We demonstrate that confining the gradient norm of loss function could help lead the optimizers towards finding flat minima. We leverage the first-order approximation to efficiently implement the corresponding gradient to fit well in the gradient descent framework. In our experiments, we confirm that when using our methods, generalization performance of various models could be improved on different datasets. Also, we show that the recent sharpness-aware minimization method  (Foretet al., 2021) is a special, but not the best, case of our method, where the best case of our method could give new state-of-art performance on these tasks.


> Poster: Disentangling Disease-related Representation from Obscure for Disease Prediction
> 
> Authors: Chu-ran Wang and Fei Gao and Fandong Zhang and Fangwei Zhong and Yizhou Yu and Yizhou Wang
> 
> Abstract: Disease-related representations play a crucial role in image-based disease prediction such as cancer diagnosis, due to its considerable generalization capacity. However, it is still a challenge to identify lesion characteristics in obscured images, as many lesions are obscured by other tissues. In this paper, to learn the representations for identifying obscured lesions, we propose a disentanglement learning strategy under the guidance of alpha blending generation in an encoder-decoder framework (DAB-Net). Specifically, we take mammogram mass benign/malignant classification as an example. In our framework, composite obscured mass images are generated by alpha blending and then explicitly disentangled into disease-related mass features and interference glands features. To achieve disentanglement learning, features of these two parts are decoded to reconstruct the mass and the glands with corresponding reconstruction losses, and only disease-related mass features are fed into the classifier for disease prediction. Experimental results on one public dataset DDSM and three in-house datasets demonstrate that the proposed strategy can achieve state-of-the-art performance. DAB-Net achieves substantial improvements of 3.9%∼4.4% AUC in obscured cases. Besides, the visualization analysis shows the model can better disentangle the mass and glands in the obscured image, suggesting the effectiveness of our solution in exploring the hidden characteristics in this challenging problem.


> Poster: FedNew: A Communication-Efficient and Privacy-Preserving Newton-Type Method for Federated Learning
> 
> Authors: Anis Elgabli and Chaouki Ben Issaid and Amrit Singh Bedi and Ketan Rajawat and Mehdi Bennis and Vaneet Aggarwal
> 
> Abstract: Newton-type methods are popular in federated learning due to their fast convergence. Still, they suffer from two main issues, namely: low communication efficiency and low privacy due to the requirement of sending Hessian information from clients to parameter server (PS). In this work, we introduced a novel framework called FedNew to deal with the above-mentioned challenges. In FedNew, there is no need to transmit Hessian information from clients to PS, hence resolving the bottleneck to improve communication efficiency. In addition, FedNew hides the gradient information and results in a privacy-preserving approach compared to the existing state-of-the-art. The core novel idea in FedNew is to introduce a two level framework, and alternate between updating the inverse Hessian-gradient product using only one alternating direction method of multipliers (ADMM) step and then performing the global model update using Newton’s method. Though only one ADMM pass is used to approximate the inverse Hessian-gradient product at each iteration, we develop a novel theoretical approach to show the convergence of FedNew to the optimal solution for strongly convex problems. Additionally, a significant reduction in communication overhead is achieved by utilizing stochastic quantization. Numerical results using real datasets show the superiority of FedNew compared to existing methods in terms of communication costs.


> Poster: Semiparametric Subgraph Reasoning for Question Answering over Large Knowledge Bases
> 
> Authors: Rajarshi Das and Ameya Godbole and Ankita Naik and Elliot Tower and Manzil Zaheer and Hannaneh Hajishirzi and Robin Jia and Andrew McCallum
> 
> Abstract: Question answering (QA) over real world knowledge bases (KBs) is challenging because of diverse, essentially unbounded, types of reasoning patterns needed. However, we hypothesize in a large KB, reasoning patterns required to answer a query type reoccur for various entities in their respective subgraph neigborhoods.Leveraging this structural similarity between local neighborhoods of different subgraphs, we introduce a semiparametric model with (i) a nonparametric component that for each query, dynamically retrieves other similar $k$-nearest neighbor (KNN) training queries along with query-specific subgraphs and (ii) a parametric component that is trained to identify the (latent) reasoning patterns from the subgraphs of KNN queries and apply it to the subgraph of the target query. We also propose a novel algorithm to select a query-specific compact subgraph from within the massive knowledge graph (KG), allowing us to scale to full Freebase KG containing billions of edges. We show that our model can answer queries requiring complex reasoning patterns more effectively than existing KG completion algorithms. The proposed model outperforms or performs competitively with state-of-the-art models on several KBQA benchmarks. 

> Poster: Kill a Bird with Two Stones: Closing the Convergence Gaps in Non-Strongly Convex Optimization by Directly Accelerated SVRG with Double Compensation and Snapshots
> 
> Authors: Yuanyuan Liu and Fanhua Shang and Weixin An and Hongying Liu and Zhouchen Lin
> 
> Abstract: Recently, some accelerated stochastic variance reduction algorithms such as Katyusha and ASVRG-ADMM achieve faster convergence than non-accelerated methods such as SVRG and SVRG-ADMM. However, there are still some gaps between the oracle complexities and their lower bounds. To fill in these gaps, this paper proposes a novel Directly Accelerated stochastic Variance reductIon (DAVIS) algorithm with two Snapshots for non-strongly convex (non-SC) unconstrained problems. Our theoretical results show that DAVIS achieves the optimal convergence rate O(1/(nS^2)) and optimal gradient complexity O(n+\sqrt{nL/\epsilon}), which is identical to its lower bound. To the best of our knowledge, this is the first directly accelerated algorithm that attains the optimal lower bound and improves the convergence rate from  O(1/S^2) to O(1/(nS^2)). Moreover, we extend DAVIS and theoretical results to non-SC problems with a structured regularizer, and prove that the proposed algorithm with double-snapshots also attains the optimal convergence rate O(1/(nS)) and optimal oracle complexity O(n+L/\epsilon) for such problems, and it is at least a factor n/S faster than existing accelerated stochastic algorithms, where n\gg S in general.


> Poster: TURF: Two-Factor, Universal, Robust, Fast Distribution Learning Algorithm
> 
> Authors: Yi Hao and Ayush Jain and Alon Orlitsky and Vaishakh Ravindrakumar
> 
> Abstract: Approximating distributions from their samples is a canonical statistical-learning problem. One of its most powerful and successful modalities approximates every distribution to an $\ell_1$ distance essentially at most a constant times larger than its closest $t$-piece degree-$d$ polynomial, where $t\ge1$ and $d\ge0$. Letting $c_{t,d}$ denote the smallest such factor, clearly $c_{1,0}=1$, and it can be shown that $c_{t,d}\ge 2$ for all other $t$ and $d$. Yet current computationally efficient algorithms show only $c_{t,1}\le 2.25$ and the bound rises quickly to $c_{t,d}\le 3$ for $d\ge 9$. We derive a near-linear-time and essentially sample-optimal estimator that establishes $c_{t,d}=2$ for all $(t,d)\ne(1,0)$. Additionally, for many practical distributions, the lowest approximation distance is achieved by polynomials with vastly varying number of pieces. We provide a method that estimates this number near-optimally, hence helps approach the best possible approximation. Experiments combining the two techniques confirm improved performance over existing methodologies.

> Poster: How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models
> 
> Authors: Ahmed Alaa and Boris van Breugel and Evgeny Saveliev and Mihaela van der Schaar
> 
> Abstract: Devising domain- and model-agnostic evaluation metrics for generative models is an important and as yet unresolved problem. Most existing metrics, which were tailored solely to the image synthesis setup, exhibit a limited capacity for diagnosing the different modes of failure of generative models across broader application domains. In this paper, we introduce a 3-dimensional evaluation metric, (α-Precision, β-Recall, Authenticity), that characterizes the fidelity, diversity and generalization performance of any generative model in a domain-agnostic fashion. Our metric unifies statistical divergence measures with precision-recall analysis, enabling sample- and distribution-level diagnoses of model fidelity and diversity. We introduce generalization as an additional, independent dimension (to the fidelity-diversity trade-off) that quantifies the extent to which a model copies training data—a crucial performance indicator when modeling sensitive data with requirements on privacy. The three metric components correspond to (interpretable) probabilistic quantities, and are estimated via sample-level binary classification. The sample-level nature of our metric inspires a novel use case which we call model auditing, wherein we judge the quality of individual samples generated by a (black-box) model, discarding low-quality samples and hence improving the overall model performance in a post-hoc manner.


> Poster: Robustness in Multi-Objective Submodular Optimization: a Quantile Approach
> 
> Authors: Cedric Malherbe and Kevin Scaman
> 
> Abstract: The optimization of multi-objective submodular systems appears in a wide variety of applications. However, there are currently very few techniques which are able to provide a robust allocation to such systems. In this work, we propose to design and analyse novel algorithms for the robust allocation of submodular systems through lens of quantile maximization. We start by observing that identifying an exact solution for this problem is computationally intractable. To tackle this issue, we propose a proxy for the quantile function using a softmax formulation, and show that this proxy is well suited to submodular optimization. Based on this relaxation, we propose a novel and simple algorithm called SOFTSAT. Theoretical properties are provided for this algorithm as well as novel approximation guarantees. Finally, we provide numerical experiments showing the efficiency of our algorithm with regards to state-of-the-art methods in a test bed of real-world applications, and show that SOFTSAT is particularly robust and well-suited to online scenarios.


> Poster: Failure and success of the spectral bias prediction for Kernel Ridge Regression: the case of low-dimensional data
> 
> Authors: Umberto Tomasini and Antonio Sclocchi and Matthieu Wyart
> 
> Abstract: Recently, several theories including the replica method made predictions for the generalization error of Kernel Ridge Regression. In some regimes, they predict that the method has a `spectral bias': decomposing the true function $f^*$ on the eigenbasis of the kernel, it fits well the coefficients associated with the O(P) largest eigenvalues, where $P$ is the size of the training set. This prediction works very well on benchmark data sets such as images, yet the assumptions these approaches make on the data are never satisfied in practice. To clarify when the spectral bias prediction holds, we first focus on a one-dimensional model where rigorous results are obtained and then use scaling arguments to generalize and test our findings in higher dimensions. Our predictions include  the classification case  $f(x)=$sign$(x_1)$ with a data distribution that vanishes at the decision boundary $p(x)\sim x_1^{\chi}$. For $\chi>0$ and a Laplace kernel, we find  that (i) there exists a cross-over ridge  $\lambda^*_{d,\chi}(P)\sim P^{-\frac{1}{d+\chi}}$ such that for $\lambda\gg \lambda^*_{d,\chi}(P)$, the replica method applies, but not for $\lambda\ll\lambda^*_{d,\chi}(P)$, (ii) in the ridge-less case, spectral bias predicts the correct training curve exponent only in the limit $d\rightarrow\infty$. 

> Poster: Differentially Private Approximate Quantiles
> 
> Authors:   and Haim Kaplan and Uri Stemmer
> 
> Abstract: In this work we study the problem of differentially private (DP) quantiles, in which given dataset $X$  and quantiles $q_1, ..., q_m \in [0,1]$, we want to output $m$ quantile estimations which are as close as possible to the true quantiles and preserve DP. We describe a simple recursive DP algorithm, which we call Approximate Quantiles (AQ), for this task. We give a worst case upper bound on its error, and show that its error is much lower than of previous implementations on several different datasets. Furthermore, it gets this low error while running time two orders of magnitude faster that the best previous implementation.

> Poster: Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning
> 
> Authors: Haoqi Yuan and Zongqing Lu
> 
> Abstract: We study offline meta-reinforcement learning, a practical reinforcement learning paradigm that learns from offline data to adapt to new tasks. The distribution of offline data is determined jointly by the behavior policy and the task. Existing offline meta-reinforcement learning algorithms cannot distinguish these factors, making task representations unstable to the change of behavior policies. To address this problem, we propose a contrastive learning framework for task representations that are robust to the distribution mismatch of behavior policies in training and test. We design a bi-level encoder structure, use mutual information maximization to formalize task representation learning, derive a contrastive learning objective, and introduce several approaches to approximate the true distribution of negative pairs. Experiments on a variety of offline meta-reinforcement learning benchmarks demonstrate the advantages of our method over prior methods, especially on the generalization to out-of-distribution behavior policies.


> Poster: Path-aware and structure-preserving generation of synthetically accessible molecules
> 
> Authors: Juhwan Noh and Dae-Woong Jeong and Kiyoung Kim and Sehui Han and Moontae Lee and Honglak Lee and Yousung Jung
> 
> Abstract: Computational chemistry aims to autonomously design specific molecules with target functionality. Generative frameworks provide useful tools to learn continuous representations of molecules in a latent space. While modelers could optimize chemical properties, many generated molecules are indeed not synthesizable. To design synthetically accessible molecules that preserve main structural motifs of target molecules, we propose a reaction-embedded and structure-conditioned variational autoencoder. As the latent space jointly encodes molecular structures and their reaction routes, our new sampling method that measures the path-informed structural similarity allows us to effectively generate structurally analogous synthesizable molecules. When targeting out-of-domain as well as in-domain seed structures, our model generates structurally and property-wisely similar molecules equipped with well-defined reaction paths. By focusing on the important region in chemical space, we also demonstrate that our model can design new molecules with even higher activity than the seed molecules.


> Poster: Cycle Representation Learning for Inductive Relation Prediction
> 
> Authors: Zuoyu Yan and Tengfei Ma and Liangcai Gao and Zhi Tang and Chao Chen
> 
> Abstract: In recent years, algebraic topology and its modern development, the theory of persistent homology, has shown great potential in graph representation learning. In this paper, based on the mathematics of algebraic topology, we propose a novel solution for inductive relation prediction, an important learning task for knowledge graph completion. To predict the relation between two entities, one can use the existence of rules, namely a sequence of relations. Previous works view rules as paths and primarily focus on the searching of paths between entities. The space of rules is huge, and one has to sacrifice either efficiency or accuracy. In this paper, we consider rules as cycles and show that the space of cycles has a unique structure based on the mathematics of algebraic topology. By exploring the linear structure of the cycle space, we can improve the searching efficiency of rules. We propose to collect cycle bases that span the space of cycles. We build a novel GNN framework on the collected cycles to learn the representations of cycles, and to predict the existence/non-existence of a relation. Our method achieves state-of-the-art performance on benchmarks.


> Poster: The Primacy Bias in Deep Reinforcement Learning
> 
> Authors: Evgenii Nikishin and Max Schwarzer and Pierluca D'Oro and Pierre-Luc Bacon and Aaron Courville
> 
> Abstract: This work identifies a common flaw of deep reinforcement learning (RL) algorithms: a tendency to rely on early interactions and ignore useful evidence encountered later. Because of training on progressively growing datasets, deep RL agents incur a risk of overfitting to earlier experiences, negatively affecting the rest of the learning process. Inspired by cognitive science, we refer to this effect as the primacy bias. Through a series of experiments, we dissect the algorithmic aspects of deep RL that exacerbate this bias. We then propose a simple yet generally-applicable mechanism that tackles the primacy bias by periodically resetting a part of the agent. We apply this mechanism to algorithms in both discrete (Atari 100k) and continuous action (DeepMind Control Suite) domains, consistently improving their performance.


> Poster: The power of first-order smooth optimization for black-box non-smooth problems
> 
> Authors: Alexander Gasnikov and Anton Novitskii and Vasilii Novitskii and Farshed Abdukhakimov and Dmitry Kamzolov and Aleksandr Beznosikov and Martin Takac and Pavel Dvurechenskii and Bin Gu
> 
> Abstract: Gradient-free/zeroth-order methods for black-box convex optimization have been extensively studied in the last decade with the main focus on oracle calls complexity. In this paper, besides the oracle complexity, we focus also on iteration complexity, and propose a generic approach that, based on optimal first-order methods, allows to obtain in a black-box fashion new zeroth-order algorithms for non-smooth convex optimization problems. Our approach not only leads to optimal oracle complexity, but also allows to obtain iteration complexity similar to first-order methods, which, in turn, allows to exploit parallel computations to accelerate the convergence of our algorithms. We also elaborate on extensions for stochastic optimization problems, saddle-point problems, and distributed optimization.


> Poster: Versatile Dueling Bandits: Best-of-both World Analyses for Learning from Relative Preferences
> 
> Authors: Aadirupa Saha and Pierre Gaillard
> 
> Abstract: We study the problem of $K$-armed dueling bandit for both stochastic and adversarial environment, where the goal of the learner is to aggregate information through relative preferences of pair of decisions points queried in an online sequential manner. We first propose a novel reduction from any (general) dueling bandits to multi-armed bandits and despite the simplicity, it allows us to improve many existing results in dueling bandits. In particular, \emph{we give the first best-of-both world result for the dueling bandits regret minimization problem}---a unified framework that is guaranteed to perform optimally for both stochastic and adversarial preferences simultaneously. Moreover, our algorithm is also the first to achieve an optimal $O(\sum_{i = 1}^K \frac{\log T}{\Delta_i})$ regret bound against the Condorcet-winner benchmark, which scales optimally both in terms of the arm-size $K$ and the instance-specific suboptimality gaps $\{\Delta_i\}_{i = 1}^K$. This resolves the long standing problem of designing an instancewise gap-dependent order optimal regret algorithm for dueling bandits (with matching lower bounds up to small constant factors). We further justify the robustness of our proposed algorithm by proving its optimal regret rate under adversarially corrupted preferences---this outperforms the existing state of the art corrupted dueling results by a large margin. The efficacy of our proposed algorithms are empirically corroborated against state of the dueling bandit methods.

> Poster: Diffusion bridges vector quantized variational autoencoders
> 
> Authors: Max Cohen and Guillaume QUISPE and Sylvain Le Corff and Charles Ollion and Eric Moulines
> 
> Abstract: Vector Quantised-Variational AutoEncoders (VQ-VAE) are generative models based on discrete latent representations of the data, where inputs are mapped to a finite set of learned embeddings. To generate new samples, an autoregressive prior distribution over the discrete states must be trained separately. This prior is generally very complex and leads to very slow generation. In this work, we propose a new model to train the prior and the encoder/decoder networks simultaneously. We build a diffusion bridge between a continuous coded vector and a non-informative prior distribution.  The latent discrete states are then given as random functions of these continuous vectors. We show that our model is competitive with the autoregressive prior on the mini-Imagenet dataset and is very efficient in both optimization and sampling. Our framework also extends the standard VQ-VAE and enables end-to-end training.


> Poster: A psychological theory of explainability
> 
> Authors: Nils Erik Tomas Folke and Scott Cheng-Hsin Yang and Patrick Shafto
> 
> Abstract: The goal of explainable Artificial Intelligence (XAI) is to generate human-interpretable explanations, but there are no computationally precise theories of how humans interpret AI generated explanations. The lack of theory means that validation of XAI must be done empirically, on a case-by-case basis, which prevents systematic theory-building in XAI. We propose a psychological theory of how humans draw conclusions from saliency maps, the most common form of XAI explanation, which for the first time allows for precise prediction of explainee inference conditioned on explanation. Our theory posits that absent explanation humans expect the AI to make similar decisions to themselves, and that they interpret an explanation by comparison to the explanations they themselves would give. Comparison is formalized via Shepard's universal law of generalization in a similarity space, a classic theory from cognitive science. A pre-registered user study on AI image classifications with saliency map explanations demonstrate that our theory quantitatively matches participants' predictions of the AI.


> Poster: Exploiting Independent Instruments: Identification and Distribution Generalization
> 
> Authors: Sorawit Saengkyongam and Leonard Henckel and Niklas Pfister and Jonas Peters
> 
> Abstract: Instrumental variable models allow us to identify a causal function between covariates $X$ and a response $Y$, even in the presence of unobserved confounding. Most of the existing estimators assume that the error term in the response $Y$ and the hidden confounders are uncorrelated with the instruments $Z$. This is often motivated by a graphical separation, an argument that also justifies independence. Posing an independence condition, however, leads to strictly stronger identifiability results. We connect to existing literature in econometrics and provide a practical method for exploiting independence that can be combined with any gradient-based learning procedure. We see that even in identifiable settings, taking into account higher moments may yield better finite sample results. Furthermore, we exploit the independence for distribution generalization. We prove that the proposed estimator is invariant to distributional shifts on the instruments and worst-case optimal whenever these shifts are sufficiently strong. These results hold even in the under-identified case where the instruments are not sufficiently rich to identify the causal function.

> Poster: Adaptive Model Design for Markov Decision Process
> 
> Authors:   and Donglin Yang and Jiayang Li and Senmiao Wang and Zhuoran Yang and Zhaoran Wang
> 
> Abstract: In a Markov decision process (MDP), the optimal policy selected by the agent is achieved by an evolutionary process by means of which it incrementally searches for better policies. During this process, the agent usually does not bear the external costs/benefits of its actions, sometimes leading to an inefficient outcome that only fulfills its own interest. Therefore, appropriate regulations are often required to induce a more desirable outcome in an MDP model. In this paper, we study how to regulate such an agent by redesigning model parameters that can affect the rewards and/or the transition kernels. We formulate this problem as a hierarchical mathematical program, in which the lower level MDP is regulated by the upper-level model designer. To solve this problem, we develop a scheme that allows the designer to iteratively predict the reaction of the agent by solving the MDP, and then adaptively update model parameters to guide the agent's behavior towards the desired end. The convergence of the algorithm is first theoretically analyzed and then empirically tested on several MDP models arising in economics and robotics.


> Poster: SDQ: Stochastic Differentiable Quantization with Mixed Precision
> 
> Authors: Xijie Huang and Zhiqiang Shen and Shichao Li and Zechun Liu and Hu Xianghong and Jeffry Wicaksana and Eric Xing and Kwang-Ting Cheng
> 
> Abstract: In order to deploy deep models in a computationally efficient manner, model quantization approaches have been frequently used. In addition, as new hardware that supports various-bit arithmetic operations, recent research on mixed precision quantization (MPQ) begins to fully leverage the capacity of representation by searching various bitwidths for different layers and modules in a network. However, previous studies mainly search the MPQ strategy in a costly scheme using reinforcement learning, neural architecture search, etc., or simply utilize partial prior knowledge for bitwidth distribution, which might be biased and sub-optimal. In this work, we present a novel Stochastic Differentiable Quantization (SDQ) method that can automatically learn the MPQ strategy in a more flexible and globally-optimized space with a smoother gradient approximation. Particularly, Differentiable Bitwidth Parameters (DBPs) are employed as the probability factors in stochastic quantization between adjacent bitwidth. After the optimal MPQ strategy is acquired, we further train our network with the entropy-aware bin regularization and knowledge distillation. We extensively evaluate our method on different networks, hardwares (GPUs and FPGA), and datasets. SDQ outperforms all other state-of-the-art mixed or single precision quantization with less bitwidth, and are even better than the original full-precision counterparts across various ResNet and MobileNet families, demonstrating the effectiveness and superiority of our method. Code will be publicly available.


> Poster: Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification
> 
> Authors:   and Amir Atapour-Abarghouei
> 
> Abstract: Convolutional Neural Networks have demonstrated dermatologist-level performance in the classification of melanoma and other skin lesions, but prediction irregularities due to biases seen within the training data are an issue that should be addressed before widespread deployment is possible. In this work, we robustly remove bias and spurious variation from an automated melanoma classification pipeline using two leading bias unlearning techniques. We show that the biases introduced by surgical markings and rulers presented in previous studies can be reasonably mitigated using these bias removal methods. We also demonstrate the generalisation benefits of unlearning spurious variation relating to the imaging instrument used to capture lesion images. Our experimental results provide evidence that the effects of each of the aforementioned biases are notably reduced, with different debiasing techniques excelling at different tasks.


> Poster: Translating Robot Skills: Learning Unsupervised Skill Correspondences Across Robots
> 
> Authors: Tanmay Shankar and Yixin Lin and Aravind Rajeswaran and Vikash Kumar and Stuart Anderson and Jean Oh
> 
> Abstract: In this paper, we explore how we can endow robots with the ability to learn correspondences between their own skills, and those of morphologically different robots in different domains, in an entirely unsupervised manner. We make the insight that different morphological robots use similar task strategies to solve similar tasks. Based on this insight, we frame learning skill correspondences as a problem of matching distributions of sequences of skills across robots. We then present an unsupervised objective that encourages a learnt skill translation model to match these distributions across domains, inspired by recent advances in unsupervised machine translation.  Our approach is able to learn semantically meaningful correspondences between skills across multiple robot-robot and human-robot domain pairs despite being completely unsupervised. Further, the learnt correspondences enable the transfer of task strategies across robots and domains. We present dynamic visualizations of our results at https://sites.google.com/view/translatingrobotskills/home. 


> Poster: DNNR: Differential Nearest Neighbors Regression
> 
> Authors: Youssef Nader and Leon Sixt and Tim Landgraf
> 
> Abstract: K-nearest neighbors (KNN) is one of the earliest and most established algorithms in machine learning. For regression tasks, KNN averages the targets within a neighborhood which poses a number of challenges: the neighborhood definition is crucial for the predictive performance as neighbors might be selected based on uninformative features, and averaging does not account for how the function changes locally. We propose a novel method called Differential Nearest Neighbors Regression (DNNR) that addresses both issues simultaneously: during training, DNNR estimates local gradients to scale the features; during inference, it performs an n-th order Taylor approximation using estimated gradients. In a large-scale evaluation on over 250 datasets, we find that DNNR performs comparably to state-of-the-art gradient boosting methods and MLPs while maintaining the simplicity and transparency of KNN. This allows us to derive theoretical error bounds and inspect failures. In times that call for transparency of ML models, DNNR provides a good balance between performance and interpretability.


> Poster: Minimax Classification under Concept Drift with Multidimensional Adaptation and Performance Guarantees
> 
> Authors: Verónica Álvarez and Santiago Mazuelas and Jose A Lozano
> 
> Abstract: The statistical characteristics describing the underlying distribution of instance-label pairs often change with time in practical scenarios of supervised classification. Conventional learning techniques adapt to such concept drift accounting for a scalar rate of change by means of a carefully chosen learning rate, forgetting factor, or window size. However, the time changes in common scenarios are multidimensional, i.e., different statistical characteristics often change in a different manner. This paper presents adaptive minimax risk classifiers (AMRCs) that account for multidimensional time changes by means of a multivariate and high-order tracking of the time- varying underlying distribution. In addition, differently from conventional techniques, AMRCs can provide computable tight performance guarantees. Experiments on multiple benchmark datasets show the classification improvement of AMRCs compared to the state-of-the-art and the reliability of the presented performance guarantees.


> Poster: AdAUC: End-to-end Adversarial AUC Optimization Against Long-tail Problems
> 
> Authors: Wenzheng Hou and Qianqian Xu and zhiyong yang and Shilong Bao and Yuan He and Qingming Huang
> 
> Abstract: It is well-known that deep learning models are vulnerable to adversarial examples. Existing studies of adversarial training have made great progress against this challenge. As a typical trait, they often assume that the class distribution is overall balanced. However, long-tail datasets are ubiquitous in a wide spectrum of applications, where the amount of head class instances is significantly larger than the tail classes. Under such a scenario, AUC is a much more reasonable metric than accuracy since it is insensitive toward class distribution. Motivated by this, we present an early trial to explore adversarial training methods to optimize AUC. The main challenge lies in that the positive and negative examples are tightly coupled in the objective function. As a direct result, one cannot generate adversarial examples without a full scan of the dataset. To address this issue, based on a concavity regularization scheme, we reformulate the AUC optimization problem as a saddle point problem, where the objective becomes an instance-wise function. This leads to an end-to-end training protocol. Furthermore, we provide a convergence guarantee of the proposed training algorithm. Our analysis differs from the existing studies since the algorithm is asked to generate adversarial examples by calculating the gradient of a min-max problem. Finally, the extensive experimental results show the performance and robustness of our algorithm in three long-tail datasets.


> Poster: 3D Infomax improves GNNs for Molecular Property Prediction
> 
> Authors: Hannes Stärk and Dominique Beaini and Gabriele Corso and Prudencio Tossou and Christian Dallago and Stephan Günnemann and Pietro Lió
> 
> Abstract: Molecular property prediction is one of the fastest-growing applications of deep learning with critical real-world impacts. Although the 3D molecular graph structure is necessary for models to achieve strong performance on many tasks, it is infeasible to obtain 3D structures at the scale required by many real-world applications. To tackle this issue, we propose to use existing 3D molecular datasets to pre-train a model to reason about the geometry of molecules given only their 2D molecular graphs. Our method, called 3D Infomax, maximizes the mutual information between learned 3D summary vectors and the representations of a graph neural network (GNN). During fine-tuning on molecules with unknown geometry, the GNN is still able to produce implicit 3D information and uses it for downstream tasks. We show that 3D Infomax provides significant improvements for a wide range of properties, including a 22% average MAE reduction on QM9 quantum mechanical properties. Moreover, the learned representations can be effectively transferred between datasets in different molecular spaces. 


> Poster: PAGE-PG: A Simple and Loopless Variance-Reduced Policy Gradient Method with Probabilistic Gradient Estimation
> 
> Authors: Matilde Gargiani and Andrea Zanelli and Andrea Martinelli and Tyler Summers and John Lygeros
> 
> Abstract: Despite their great success, policy gradient methods suffer from the high variance of the gradient estimate, which generally results in a bad sample complexity. Recently, numerous variance-reduced extensions of policy gradient methods with provably better sample complexity and competitive numerical performance have been proposed. Variance-reduction techniques were successfully deployed in supervised learning long before their adaptation to reinforcement learning, which required to account for different challenges, such as the distribution shift and the infinite dataset. In this work, after conducting a compact survey on some of the main variance-reduced REINFORCE-type methods, we propose a novel loopless variance-reduced policy gradient method, ProbAbilistic Gradient Estimation for Policy Gradient (PAGE-PG). Our method is inspired by the PAGE estimator for supervised learning and leverages importance sampling to obtain an unbiased gradient estimator. We show that PAGE-PG enjoys a $\mathcal{O}\left( \epsilon^{-3} \right)$ average sample complexity to reach an $\epsilon$-accurate solution, which matches the sample complexity of state-of-the-art variance-reduced policy gradient methods under the same setting. The novelty of PAGE-PG consists in replacing the double loop structure typical of variance-reduced methods with a probabilistic switch between two types of updates. The probabilistic switch allows for a truly loopless structure, which facilitates the theoretical analysis and hyper-parameter tuning. In addition, it lends itself to noise annealing strategies that favor exploration in the early stages of training as well as to more sophisticated strategies that adaptively regulate the noise level to improve convergence in presence of complex nonconcave landscapes. Finally, the numerical evaluation confirms the competitive performance of our method on different control tasks.  

> Poster: Equivariant graph neural networks with complete local frames
> 
> Authors: weitao du and He Zhang and Yuanqi Du and Qi Meng and Wei Chen and Tie-Yan Liu and Nanning Zheng and Bin Shao
> 
> Abstract: Group equivariance (e.g. SE(3) equivariance) is a critical physical symmetry in science, from classical and quantum physics to computational biology. It enables robust and accurate prediction under arbitrary reference transformations. In light of this, great efforts have been put on encoding this symmetry into deep neural networks, which has been shown to improve the generalization performance and data efficiency for downstream tasks.  Constructing an equivariant neural network generally brings high computational costs to ensure expressiveness. Therefore, how to better trade-off the expressiveness and computational efficiency plays a core role in the design of the equivariant deep learning models.  In this paper, we propose a framework to construct SE(3) equivariant graph neural networks that can approximate the geometric quantities efficiently. Inspired by differential geometry and physics, we introduce equivariant local complete frames to graph neural networks, such that tensor information at given orders can be projected onto the frames. The local frame is constructed to form an orthonormal basis that avoids direction degeneration and ensure completeness. Since the frames are built only by cross product operations, our method is computationally efficient. %(e.g., compared with the Clebsch-Gordan tensor product). We evaluate our method on two tasks: Newton mechanics modeling and equilibrium molecule conformation generation. Extensive experimental results demonstrate that our model achieves the best or competitive performance in various types of datasets.


> Poster: Least Squares Estimation using Sketched Data with Heteroskedastic Errors
> 
> Authors: Sokbae Lee and Serena Ng
> 
> Abstract: Researchers may use a sketch of data of size m instead of the full sample of size n sometimes to relieve computation burden, and other times to maintain data privacy. This paper considers the case when full sample estimation would have required the White-Eicker robust standard errors to account for heteroskedasticity. We show random projections have a smoothing effect on the sketched data, and the least squares estimates using sketched data behave 'as if' the errors were homoskedastic. This result is obtained by expressing the difference between the moments computed from the full sample and the sketched data as a degenerate U statistic, and such statistics are asymptotically normality with a homoskedastic variance when the conditions in Hall (1984) are satisfied. This interesting result also holds for two-stage least squares for which algorithmic and statistical properties are also analyzed. Sketches produced by random sampling do not, however, have this property.


> Poster: Principled Knowledge Extrapolation with GANs
> 
> Authors: Ruili Feng and Jie Xiao and Kecheng Zheng and Deli Zhao and Jingren Zhou and Qibin Sun and Zheng-Jun Zha
> 
> Abstract: Human can extrapolate well, generalize daily knowledge into unseen scenarios, raise and answer counterfactual questions. To imitate this ability via generative models, previous works have extensively studied explicitly encoding Structural Causal Models (SCMs) into architectures of generator networks. This methodology, however, limits the flexibility of the generator as they must be carefully crafted to follow the causal graph, and demands a ground truth SCM with strong ignorability assumption as prior, which is a nontrivial assumption in many real scenarios. Thus, many current causal GAN methods fail to generate high fidelity counterfactual results as they cannot easily leverage state-of-the-art generative models. In this paper, we propose to study counterfactual synthesis from a new perspective of knowledge extrapolation, where a given knowledge dimension of the data distribution is extrapolated, but the remaining knowledge is kept indistinguishable from the original distribution. We show that an adversarial game with a closed-form discriminator can be used to address the knowledge extrapolation problem, and a novel principal knowledge descent method can efficiently estimate the extrapolated distribution through the adversarial game. Our method enjoys both elegant theoretical guarantees and superior performance in many scenarios.


> Poster: RECAPP: Crafting a More Efficient Catalyst for Convex Optimization
> 
> Authors: Yair Carmon and Arun Jambulapati and Yujia Jin and Aaron Sidford
> 
> Abstract: The accelerated proximal point method (APPA), also known as ``Catalyst'', is a well-established reduction from convex optimization to approximate proximal point computation (i.e., regularized minimization). This reduction is conceptually elegant and yields strong convergence rate guarantees. However, these rates feature an extraneous logarithmic term arising from the need to compute each proximal point to high accuracy. In this work, we propose a novel Relaxed Error Criterion for Accelerated Proximal Point (RECAPP) that eliminates the need for high accuracy subproblem solutions. We apply RECAPP to two canonical problems: finite-sum and max-structured minimization. For finite-sum problems, we match the best known complexity, previously obtained by carefully designed problem-specific algorithms. For minimizing max_y f(x,y) where f is convex in x and strongly-concave in y, we improve on the best known bound by a logarithmic factor.


> Poster: Quantifying and Learning Linear Symmetry-Based Disentanglement
> 
> Authors: Loek Tonnaer and Luis Armando Perez Rey and Vlado Menkovski and Mike Holenderski and Jacobus Portegies
> 
> Abstract: The definition of Linear Symmetry-Based Disentanglement (LSBD) formalizes the notion of linearly disentangled representations, but there is currently no metric to quantify LSBD. Such a metric is crucial to evaluate LSBD methods and to compare them to previous understandings of disentanglement. We propose D_LSBD, a mathematically sound metric to quantify LSBD, and provide a practical implementation for SO(2) groups. Furthermore, from this metric we derive LSBD-VAE, a semi-supervised method to learn LSBD representations. We demonstrate the utility of our metric by showing that (1) common VAE-based disentanglement methods don't learn LSBD representations, (2) LSBD-VAE, as well as other recent methods, can learn LSBD representations needing only limited supervision on transformations, and (3) various desirable properties expressed by existing disentanglement metrics are also achieved by LSBD representations.


> Poster: Asymptotically-Optimal Gaussian Bandits with Side Observations
> 
> Authors: Alexia Atsidakou and Orestis Papadigenopoulos and Constantine Caramanis and Sujay Sanghavi and Sanjay Shakkottai
> 
> Abstract: We study the problem of Gaussian bandits with general side information, as first introduced by Wu, Szepesv\'{a}ri, and Gy\"{o}rgy. In this setting, the play of an arm reveals information about other arms, according to an arbitrary {\em a priori} known {\em side information} matrix: each element of this matrix encodes the fidelity of the information that the row" arm reveals about thecolumn" arm. In the case of Gaussian noise, this model subsumes standard bandits, full-feedback, and graph-structured feedback as special cases. In this work, we first construct an LP-based asymptotic instance-dependent lower bound on the regret. The LP optimizes the cost (regret) required to reliably estimate the suboptimality gap of each arm. This LP lower bound motivates our main contribution: the first known asymptotically optimal algorithm for this general setting. 


> Poster: Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs
> 
> Authors: Meyer Scetbon and Gabriel Peyré and Marco Cuturi
> 
> Abstract: The ability to align points across two related yet incomparable point clouds (e.g. living in different spaces) plays an important role in machine learning. The Gromov-Wasserstein (GW) framework provides an increasingly popular answer to such problems, by seeking a low-distortion, geometry-preserving assignment between these points.As a non-convex, quadratic generalization of optimal transport (OT), GW is NP-hard. While practitioners often resort to solving GW approximately as a nested sequence of entropy-regularized OT problems, the cubic complexity (in the number $n$ of samples) of that approach is a roadblock.We show in this work how a recent variant of the OT problem that restricts the set of admissible couplings to those having a low-rank factorization is remarkably well suited to the resolution of GW:when applied to GW, we show that this approach is not only able to compute a stationary point of the GW problem in time $O(n^2)$, but also uniquely positioned to benefit from the knowledge that the initial cost matrices are low-rank, to yield a linear time $O(n)$ GW approximation. Our approach yields similar results, yet orders of magnitude faster computation than the SoTA entropic GW approaches, on both simulated and real data. 

> Poster: Risk-Averse No-Regret Learning in Online Convex Games
> 
> Authors: Zifan Wang and Yi Shen and Michael Zavlanos
> 
> Abstract: We consider an online stochastic game with risk-averse agents whose goal is to learn optimal decisions that minimize the risk of incurring significantly high costs. Specifically, we use the Conditional Value at Risk (CVaR) as a risk measure that the agents can estimate using bandit feedback in the form of the cost values of only their selected actions.Since the distributions of the cost functions depend on the actions of all agents that are generally unobservable, they are themselves unknown and, therefore, the CVaR values of the costs are difficult to compute.To address this challenge, we propose a new online risk-averse learning algorithm that relies on one-point zeroth-order estimation of the CVaR gradients computed using CVaR values that are estimated by appropriately sampling the cost functions.We show that this algorithm achieves sub-linear regret with high probability. We also propose two variants of this algorithm that improve performance. The first variant relies on a new sampling strategy that uses samples from the previous iteration to improve the estimation accuracy of the CVaR values. The second variant employs residual feedback that uses CVaR values from the previous iteration to reduce the variance of the CVaR gradient estimates. We theoretically analyze the convergence properties of these variants and illustrate their performance on an online market problem that we model as a Cournot game. 


> Poster: Controlling Conditional Language Models without Catastrophic Forgetting
> 
> Authors: Tomasz Korbak and Hady Elsahar and German Kruszewski and Marc Dymetman
> 
> Abstract: Machine learning is shifting towards general-purpose pretrained generative models, trained in a self-supervised manner on large amounts of data, which can then be applied to solve a large number of tasks. However, due to their generic training methodology, these models often fail to meet some of the downstream requirements (e.g., hallucinations in abstractive summarization or wrong format in automatic code generation). This raises the important question of how to adapt pre-trained generative models to meet all requirements without destroying their general capabilities (``catastrophic forgetting'').  Recent work has proposed to solve this problem by representing task-specific requirements through energy-based models (EBMs) and approximating these EBMs using distributional policy gradients (DPG). Despite its effectiveness, this approach is however limited to unconditional distributions. In this paper, we extend DPG to conditional tasks by proposing Conditional DPG (CDPG). We evaluate CDPG on three different control objectives across two tasks: summarization with T5 and code generation with GPT-Neo. Our results show that fine-tuning using CDPG robustly moves these pretrained models closer towards meeting control objectives and --- in contrast with baseline approaches --- does not result in catastrophic forgetting.


> Poster: Auxiliary Learning with Joint Task and Data Scheduling
> 
> Authors: Hong Chen and Xin Wang and Chaoyu Guan and Yue Liu and wenwu zhu
> 
> Abstract: Existing auxiliary learning approaches only consider the relationships between the target task and the auxiliary tasks, ignoring the fact that data samples within an auxiliary task could contribute differently to the target task, which results in inefficient auxiliary information usage and non-robustness to data noise. In this paper, we propose to learn a joint task and data schedule for auxiliary learning, which captures the importance of different data samples in each auxiliary task to the target task. However, learning such a joint schedule is quite challenging due to the large number of additional parameters required for the schedule. To tackle the challenge, we propose a joint task and data scheduling (JTDS) model for auxiliary learning. The JTDS model captures the joint task-data importance through a parameter-efficient task-data scheduler, which creates a mapping from task, feature and label information to the schedule in a parameter-efficient manner. Particularly, we formulate the scheduler learning and the task learning process as a bi-level optimization problem. In the lower optimization, the task learning model is updated with the scheduled gradient , while in the upper optimization, the task-data scheduler is updated with implicit gradient from a developing dataset. Experimental results show that our proposed JTDS model significantly outperforms the state-of-the-art methods under supervised learning, semi-supervised learning and corrupted label settings.  


> Poster: Data Determines Distributional Robustness in Contrastive Language Image Pre-training
> 
> Authors: Alex Fang and Vaishaal Shankar and Achal Dave and Yuhao Wan and Gabriel Ilharco and Mitchell Wortsman and Ludwig Schmidt
> 
> Abstract: Contrastively trained image-text models such as CLIP, ALIGN, and BASIC have demonstrated unprecedented robustness to multiple challenging natural distribution shifts. Since these image-text models differ from previous training approaches in several ways, an important question is what causes the large robustness gains.We answer this question via a systematic experimental investigation.Concretely, we study five different possible causes for the robustness gains: (i) the training set size, (ii) the training distribution, (iii) language supervision at training time, (iv) language supervision at test time, and (v) the contrastive loss function. Our experiments show that the more diverse training distribution is the main cause for the robustness gains, with the other factors contributing little to no robustness. Beyond our experimental results, we also introduce ImageNet-Captions, a version of ImageNet with original text annotations from Flickr, to enable further controlled experiments of language-image training.


> Poster: Sparse Double Descent: Where Network Pruning Aggravates Overfitting
> 
> Authors: Zheng He and Zeke Xie and Quanzhi Zhu and Zengchang Qin
> 
> Abstract: People usually believe that network pruning not only reduces the computational cost of deep networks, but also prevents overfitting by decreasing model capacity. However, our work surprisingly discover that moderately sparse models sometimes even aggravate overfitting, which is in direct contradiction to the conventional wisdom on network pruning. We report an unexpected sparse double descent phenomenon that, as we increase model sparsity via network pruning, test performance first gets worse (due to overfitting), then gets better (due to relieved overfitting), and gets worse at last (due to forgetting useful information). While recent studies focused on the deep double descent with respect to model overparameterization, they failed to recognize that sparsity may also cause the double descent. In this paper, we not only demonstrate the existence and ubiquity of sparse double descent through extensive experiments, but also propose a novel learning distance interpretation for this phenomenon. We suggest that the curve of ℓ2 learning distance of sparse models (from initialized parameters to final parameters) may exhibit a double descent, and reflects generalization better than minima flatness.


> Poster: Improving Self-Supervised Speech Representations by Disentangling Speakers
> 
> Authors: Kaizhi Qian and Yang Zhang and Heting Gao and Junrui Ni and Cheng-I Lai and David Cox and Mark Hasegawa-Johnson and Shiyu Chang
> 
> Abstract: Self-supervised learning in speech involves training a speech representation network on a large-scale unannotated speech corpus, and then applying the learned representations to downstream tasks. Since the majority of the downstream tasks of SSL learning in speech largely focus on the content information in speech, the most desirable speech representations should be able to disentangle unwanted variations, such as speaker variations, from the content. However, disentangling speakers is very challenging, because removing the speaker information could easily result in a loss of content as well, and the damage of the latter usually far outweighs the benefit of the former. In this paper, we propose a new SSL method that can achieve speaker disentanglement without severe loss of content. Our approach is adapted from the HuBERT framework, and incorporates disentangling mechanisms to regularize both the teacher labels and the learned representations. We evaluate the benefit of speaker disentanglement on a set of content-related downstream tasks, and observe a consistent and notable performance advantage of our speaker-disentangled representations.


> Poster: Strategies for Safe Multi-Armed Bandits with Logarithmic Regret and Risk
> 
> Authors: Tianrui Chen and Aditya Gangrade and Venkatesh Saligrama
> 
> Abstract: We investigate a natural but surprisingly unstudied approach to the multi-armed bandit problem under safety risk constraints. Each arm is associated with an unknown law on safety risks and rewards, and the learner's goal is to maximise reward whilst not playing unsafe arms, as determined by a given threshold on the mean risk.We formulate a pseudo-regret for this setting that enforces this safety constraint in a per-round way by softly penalising any violation, regardless of the gain in reward due to the same. This has practical relevance to scenarios such as clinical trials, where one must maintain safety for each round rather than in an aggregated sense.We describe doubly optimistic strategies for this scenario, which maintain optimistic indices for both safety risk and reward. We show that schema based on both frequentist and Bayesian indices satisfy tight gap-dependent logarithmic regret bounds, and further that these play unsafe arms only logarithmically many times in total. This theoretical analysis is complemented by simulation studies demonstrating the effectiveness of the proposed schema, and probing the domains in which their use is appropriate.


> Poster: IDYNO: Learning Nonparametric DAGs from Interventional Dynamic Data
> 
> Authors: Tian Gao and DEBARUN BHATTACHARJYA and Elliot Nelson and Miao Liu and Yue Yu
> 
> Abstract: Causal discovery in the form of a directed acyclic graph (DAG) for time series data has been widely studied in various domains. The resulting DAG typically represents a dynamic Bayesian network (DBN), capturing both the instantaneous and time-delayed relationships among variables of interest. We propose a new algorithm,  IDYNO, to learn the DAG structure from potentially  nonlinear times series data by using a continuous optimization framework that includes a recent formulation for continuous acyclicity  constraint. The proposed algorithm is designed to handle both observational and interventional time series data. We demonstrate the promising performance of our method on synthetic benchmark datasets against state-of-the-art baselines. In addition, we show that the proposed method can more accurately learn the underlying structure of a sequential decision model, such as a Markov decision process, with a fixed policy in typical continuous control tasks.


> Poster: Disentangling Sources of Risk for Distributional Multi-Agent Reinforcement Learning
> 
> Authors: Kyunghwan Son and Junsu Kim and   and Roben Delos Reyes and Yung Yi and Jinwoo Shin
> 
> Abstract: In cooperative multi-agent reinforcement learning, the outcomes of agent-wise policies are highly stochastic due to the two sources of risk: (a) random actions taken by teammates and (b) random transition and rewards. Although the two sources have very distinct characteristics, existing frameworks are insufficient to control the risk-sensitivity of agent-wise policies in a disentangled manner. To this end, we propose Disentangled RIsk-sensitive Multi-Agent reinforcement learning (DRIMA) to separately access the risk sources. For example, our framework allows an agent to be optimistic with respect to teammates (who can prosocially adapt) but more risk-neutral with respect to the environment (which does not adapt). Our experiments demonstrate that DRIMA significantly outperforms prior state-of-the-art methods across various scenarios in the StarCraft Multi-agent Challenge environment. Notably, DRIMA shows robust performance where prior methods learn only a highly suboptimal policy, regardless of reward shaping, exploration scheduling, and noisy (random or adversarial) agents.


> Poster: Unsupervised Image Representation Learning with Deep Latent Particles
> 
> Authors: Tal Daniel and Aviv Tamar
> 
> Abstract: We propose a new representation of visual data that disentangles object position from appearance. Our method, termed Deep Latent Particles (DLP), decomposes the visual input into low-dimensional latent ``particles'', where each particle is described by its spatial location and features of its surrounding region. To drive learning of such representations, we follow a VAE-based based approach and introduce a prior for particle positions based on a spatial-Softmax architecture, and a modification of the evidence lower bound loss inspired by the Chamfer distance between particles. We demonstrate that our DLP representations are useful for downstream tasks such as unsupervised keypoint (KP) detection, image manipulation, and video prediction for scenes composed of multiple dynamic objects. In addition, we show that our probabilistic interpretation of the problem naturally provides uncertainty estimates for particle locations, which can be used for model selection, among other tasks.


> Poster: Unraveling Attention via Convex Duality: Analysis and Interpretations of Vision Transformers
> 
> Authors: Arda Sahiner and Tolga Ergen and Batu M Ozturkler and John Pauly and Morteza Mardani and Mert Pilanci
> 
> Abstract: Vision transformers using self-attention or its proposed alternatives have demonstrated promising results in many image related tasks. However, the underpinning inductive bias of attention is not well understood. To address this issue, this paper analyzes attention through the lens of convex duality. For the non-linear dot-product self-attention, and alternative mechanisms such as MLP-mixer and Fourier Neural Operator (FNO), we derive equivalent finite-dimensional convex problems that are interpretable and solvable to global optimality. The convex programs lead to block nuclear-norm regularization that promotes low rank in the latent feature and token dimensions. In particular, we show how self-attention networks implicitly clusters the tokens, based on their latent similarity. We conduct experiments for transferring a pre-trained transformer backbone for CIFAR-100 classification by fine-tuning a variety of convex attention heads. The results indicate the merits of the bias induced by attention compared with the existing MLP or linear heads.


> Poster: Distributionally-Aware Kernelized Bandit Problems for Risk Aversion
> 
> Authors: Sho Takemori
> 
> Abstract: The kernelized bandit problem is a theoretically justified framework and has solid applications to various fields. Recently, there is a growing interest in generalizing the problem to the optimization of risk-averse metrics such as Conditional Value-at-Risk (CVaR) or Mean-Variance (MV).However, due to the model assumption, most existing methods need explicit design of environment random variables and can incur large regret because of possible high dimensionality of them.To address the issues, in this paper, we model environments using a family of the output distributions (or more precisely, probability kernel) and Kernel Mean Embeddings (KME), and provide novel UCB-type algorithms for CVaR and MV.Moreover, we provide algorithm-independent lower bounds for CVaR in the case of Mat\'ern kernels, and propose a nearly optimal algorithm.Furthermore, we empirically verify our theoretical result in synthetic environments, and demonstrate that our proposed method significantly outperforms a baseline in many cases.


> Poster: Transformer Quality in Linear Time
> 
> Authors: Weizhe Hua and Zihang Dai and Hanxiao Liu and Quoc Le
> 
> Abstract: We challenge some key design choices in Transformers by presenting a simpler yet more powerful layer named gated attention unit. The layer offers several desirable properties that enable a high-performance, accelerator-friendly approximate attention mechanism. The resulting architecture family, named FLAC, simultaneously achieves Transformer quality and linear cost over a wide range of context lengths (from 512 to 8K). Experiments on bidirectional and auto-regressive language modeling tasks demonstrate that FLAC can compete with fully augmented Transformers in quality, while being substantially faster to train than existing efficient attention methods.


> Poster: Multi-Level Branched Regularization for Federated Learning
> 
> Authors: Jinkyu Kim and Geeho Kim and Bohyung Han
> 
> Abstract: A critical challenge of federated learning is data heterogeneity and imbalance across clients, which leads to inconsistency between local networks and unstable convergence of global models.To alleviate the limitation, we propose a novel regularization technique, which stems a branch from each level in a local model and augments a matching subnetwork of the global model.The branch facilitates learning the representations congruent to the main pathway of the local model, leading to consistency with the global model eventually.Our regularization method is unique in the sense that it does not attempt to preserve the model parameters directly while it introduces an additional constraint that aligns the logit of each branch to that of the main local branch.The proposed technique is applicable to various federated learning algorithms without extra communication costs. We perform comprehensive empirical studies on real data under various settings and demonstrate the remarkable performance of the proposed method in terms of accuracy and efficiency compared to existing methods.


> Poster: Confidence Score for Source-Free Unsupervised Domain Adaptation
> 
> Authors: Jonghyun Lee and Dahuin Jung and Junho Yim and Sungroh Yoon
> 
> Abstract: Source-free unsupervised domain adaptation (SFUDA) aims to obtain high performance in the unlabeled target domain using the pre-trained source model, not the source data.Existing SFUDA methods assign the same importance to all target samples, which is vulnerable to incorrect pseudo-labels.To differentiate between sample importance, in this study, we propose a novel sample-wise confidence score, the Joint Model-Data Structure (JMDS) score for SFUDA.Unlike existing confidence scores that use only one of the source or target domain knowledge, the JMDS score uses both knowledge.We then propose a Confidence score Weighting Adaptation using the JMDS (CoWA-JMDS) framework for SFUDA.CoWA-JMDS consists of the JMDS scores as sample weights and weight Mixup that is our proposed variant of Mixup.Weight Mixup promotes the model make more use of the target domain knowledge.The experimental results show that the JMDS score outperforms the existing confidence scores.Moreover, CoWA-JMDS achieves state-of-the-art performance on various SFUDA scenarios: closed, open, and partial-set scenarios.


> Poster: Scalable Deep Gaussian Markov Random Fields for General Graphs
> 
> Authors: Joel Oskarsson and Per Sidén and Fredrik Lindsten
> 
> Abstract: Machine learning methods on graphs have proven useful in many applications due to their ability to handle generally structured data. The framework of Gaussian Markov Random Fields (GMRFs) provides a principled way to define Gaussian models on graphs by utilizing their sparsity structure. We propose a flexible GMRF model for general graphs built on the multi-layer structure of Deep GMRFs, originally proposed for lattice graphs only. By designing a new type of layer we enable the model to scale to large graphs. The layer is constructed to allow for efficient training using variational inference and existing software frameworks for Graph Neural Networks. For a Gaussian likelihood, close to exact Bayesian inference is available for the latent field. This allows for making predictions with accompanying uncertainty estimates. The usefulness of the proposed model is verified by experiments on a number of synthetic and real world datasets, where it compares favorably to other both Bayesian and deep learning methods. 


> Poster: Interactive Inverse Reinforcement Learning for Cooperative Games
> 
> Authors: Thomas Kleine Büning and Anne-Marie George and Christos Dimitrakakis
> 
> Abstract: We study the problem of designing autonomous agents that can learn to cooperate effectively with a potentially suboptimal partner while having no access to the joint reward function. This problem is modeled as a cooperative episodic two-agent Markov decision process. We assume control over only the first of the two agents in a Stackelberg formulation of the game, where the second agent is acting so as to maximise expected utility given the first agent's policy. How should the first agent act in order to learn the joint reward function as quickly as possible and so that the joint policy is as close to optimal as possible? We analyse how knowledge about the reward function can be gained in this interactive two-agent scenario. We show that when the learning agent's policies have a significant effect on the transition function, the reward function can be learned efficiently.


> Poster: Multiple-Play Stochastic Bandits with Shareable Finite-Capacity Arms
> 
> Authors: Xuchuang Wang and Hong Xie and John C. S. Lui
> 
> Abstract: We generalize the multiple-play multi-armed bandits (MP-MAB) problem with a shareable arms setting, in which several plays can share the same arm. Furthermore, each shareable arm has a finite reward capacity and a “per-load” reward distribution, both of which are unknown to the learner. The reward from a shareable arm is load-dependent, which is the “per-load” reward multiplying either the number of plays pulling the arm, or its reward capacity when the number of plays exceeds the capacity limit. When the “per-load” reward follows a Gaussian distribution, we prove a sample complexity lower bound of learning the capacity from load-dependent rewards and also a regret lower bound of this new MP-MAB problem. We devise a capacity estimator whose sample complexity upper bound matches the lower bound in terms of reward means and capacities. We also propose an online learning algorithm to address the problem and prove its regret upper bound. This regret upper bound's first term is the same as regret lower bound's, and its second and third terms also evidently correspond to lower bound's. Extensive experiments validate our algorithm’s performance and also its gain in 5G & 4G base station selection.


> Poster: Distributionally Robust $Q$-Learning
> 
> Authors: Zijian Liu and Zhengqing Zhou and Perry Dong and Jerry Bai and Jose Blanchet and Wei Xu and Zhengyuan Zhou
> 
> Abstract: Reinforcement learning (RL) has demonstrated remarkable achievements in simulated environments. However, carrying this success to real environments requires the important attribute of robustness, which the existing RL algorithms often lack as they assume the future deployment environment is the same as the training environment (i.e. simulator) in which the policy is learned, an assumption that 1) often does not hold due to the discrepancy between the simulator and the real environment and 2) renders the learned policy fragile as a result.In this paper, we aim to make initial progress in addressing the robustness problem. In particular, we propose a novel distributionally robust $Q$-learning algorithm that learns the best policy in the worst distributional perturbation of the environment. Our algorithm first transforms the infinite-dimensional learning problem (since the environment MDP perturbation lies in an infinite-dimensional space) into a finite-dimensional dual problem and subsequently uses a multi-level Monte-Carlo scheme to approximate the dual value using samples from the simulator. Despite the complexity, we show that the resulting distributionally robust $Q$-learning algorithm asymptotically converges to optimal worst-case policy, thus making it robust to future environment changes. Simulation results further demonstrate its empirical robustness. 

> Poster: Consistent Polyhedral Surrogates for Top-k Classification and Variants
> 
> Authors: Anish Thilagar and Rafael Frongillo and Jessica Finocchiaro and Emma Goodwill
> 
> Abstract: Top-k classification is a generalization of multiclass classification used widely in information retrieval, image classification, and other extreme classification settings. Several hinge-like (piecewise linear) surrogates have been proposed for the problem, yet all are either non-convex or inconsistent. For the proposed hinge-like surrogates that are convex (i.e., polyhedral), we apply the recent embedding framework of Finocchiaro et al. (2019) to determine the prediction problem for which the surrogate is consistent. These problems can all be interpreted as variants of top-k classification, which may be better aligned with some applications. We leverage this analysis to derive constraints on the conditional label distributions under which these proposed surrogates become consistent for top-k. It has been further suggested that every convex hinge-like surrogate must be inconsistent for top-k. Yet, we use the same embedding framework to give the first consistent polyhedral surrogate for this problem.


> Poster: Thompson Sampling for Robust Transfer in Multi-Task Bandits
> 
> Authors: Zhi Wang and Chicheng Zhang and Kamalika Chaudhuri
> 
> Abstract: We study the problem of online multi-task learning where the tasks are performed within similar but not necessarily identical multi-armed bandit environments. In particular, we study how a learner can improve its overall performance across multiple related tasks through robust transfer of knowledge. While an upper confidence bound (UCB)-based algorithm has recently been shown to achieve nearly-optimal performance guarantees in a setting where all tasks are solved concurrently, it remains unclear whether Thompson sampling (TS) algorithms, which have superior empirical performance in general, share similar theoretical properties. In this work, we present a TS-type algorithm for a more general online multi-task learning protocol, which extends the concurrent setting. We provide its frequentist analysis and prove that it is also nearly-optimal using a novel concentration inequality for multi-task data aggregation at random stopping times. Finally, we evaluate the algorithm on synthetic data and show that the TS-type algorithm enjoys superior empirical performance in comparison with the UCB-based algorithm and a baseline algorithm that performs TS for each individual task without transfer.


> Poster: Topology-aware Generalization of Decentralized SGD
> 
> Authors: Tongtian Zhu and Fengxiang He and Lan Zhang and Zhengyang Niu and Mingli Song and Dacheng Tao
> 
> Abstract: This paper studies the algorithmic stability and generalizability of decentralized stochastic gradient descent (D-SGD). We prove that the consensus model learned by D-SGD is $\mathcal{O}{(m/N\unaryplus1/m\unaryplus\lambda^2)}$-stable in expectation, where $N$ is the total sample size of the whole system, $m$ is the worker number, and $\lambda$ is the spectral gap that measures the connectivity of the communication topology. These results then deliver an $\mathcal{O}{((1\unaryplus(n\lambda^2)^{\alpha/2}\unaryplus(n^2/m)^{\alpha/2})/N)}$ in-average generalization bound, characterizing the gap between the training performance and the test performance. Our bound is non-vacuous even when the spectral gap $\lambda$ is closed to $1$, which was suggested vacuous in existing literature on the projected version of D-SGD. Our theory suggests that the generalizability of D-SGD has a positive correlation with the spectral gap. Experiments of VGG-11 and ResNet-18 on CIFAR-10 and Tiny-ImageNet justify our theory. Our code will be released publicly. To our best knowledge, this is the first work on the topology-aware generalization of vanilla D-SGD.

> Poster: End-to-End Balancing for Causal Continuous Treatment-Effect Estimation
> 
> Authors: Mohammad Taha Bahadori and Eric Tchetgen Tchetgen and David Heckerman
> 
> Abstract: We study the problem of observational causal inference with continuous treatment. We focus on the challenge of estimating the causal response curve for infrequently-observed treatment values.We design a new algorithm based on the framework of entropy balancing which learns weights that directly maximize causal inference accuracy using end-to-end optimization. Our weights can be customized for different datasets and causal inference algorithms. We propose a new theory for consistency of entropy balancing for continuous treatments. Using synthetic and real-world data, we show that our proposed algorithm outperforms the entropy balancing in terms of causal inference accuracy.


> Poster: Personalized Federated Learning through Local Memorization
> 
> Authors: Othmane MARFOQ and Giovanni Neglia and Richard Vidal and Laetitia Kameni
> 
> Abstract: Federated learning allows clients to collaboratively learn statistical models while keeping their data local. Federated learning was originally used to train a unique global model to be served to all clients, but this approach might be sub-optimal when clients' local data distributions are heterogeneous. In order to tackle this limitation, recent personalized federated learning methods train a separate model for each client while still leveraging the knowledge available at other clients. In this work, we exploit the ability of deep neural networks to extract high quality vectorial representations (embeddings) from non-tabular data, e.g., images and text, to propose a personalization mechanism based on local memorization. Personalization is obtained by interpolating a collectively trained global model with a local $k$-nearest neighbors (kNN) model based on the shared representation provided by the global model. We provide generalization bounds for the proposed approach and we show on a suite of federated datasets that this approach achieves significantly higher accuracy and fairness than state-of-the-art methods.

> Poster: Stochastic Continuous Submodular Maximization: Boosting via Non-oblivious Function
> 
> Authors: Qixin Zhang and Zengde Deng and Zaiyi Chen and Haoyuan Hu and Yu Yang
> 
> Abstract: In this paper, we revisit Stochastic Continuous Submodular Maximization in both offline and online settings, which can benefit wide applications in machine learning and operations research areas. We present a boosting framework covering gradient ascent and online gradient ascent. The fundamental ingredient of our methods is a novel non-oblivious function $F$ derived from a factor-revealing optimization problem, whose any stationary point provides a  $(1-e^{-\gamma})$-approximation to the global maximum of the $\gamma$-weakly DR-submodular objective function $f\in C^{1,1}_L(\mathcal{X})$. Under the offline scenario, we propose a boosting gradient ascent method achieving $(1-e^{-\gamma}-\epsilon^{2})$-approximation after $O(1/\epsilon^2)$ iterations, which improves the $(\frac{\gamma^2}{1+\gamma^2})$ approximation ratio of the classical gradient ascent algorithm.In the online setting, for the first time we consider the adversarial delays for stochastic gradient feedback, under which we propose a boosting online gradient algorithm with the same non-oblivious function $F$. Meanwhile, we verify that this boosting online algorithm achieves a regret of $O(\sqrt{D})$ against a $(1-e^{-\gamma})$-approximation to the best feasible solution in hindsight, where $D$ is the sum of delays of gradient feedback. To the best of our knowledge, this is the first result to obtain $O(\sqrt{T})$ regret against a $(1-e^{-\gamma})$-approximation with $O(1)$ gradient inquiry at each time step, when no delay exists, i.e., $D=T$. Finally, numerical experiments demonstrate the effectiveness of our boosting methods.

> Poster: DisPFL: Towards Communication-Efficient Personalized Federated learning via Decentralized Sparse Training
> 
> Authors: Rong Dai and Li Shen and Fengxiang He and Xinmei Tian and Dacheng Tao
> 
> Abstract: Personalized federated learning is proposed to handle the data heterogeneity problem amongst clients by learning dedicated tailored local models for each user. However, existing works are often built in a centralized way, leading to high communication pressure and high vulnerability when a failure or an attack on the central server occurs. In this work, we propose a novel personalized federated learning framework in a decentralized (peer-to-peer) communication protocol named DisPFL, which employs personalized sparse masks to customize sparse local models on the edge. To further save the communication and computation cost, we propose a decentralized sparse training technique, which means that each local model in DisPFL only maintains a fixed number of active parameters throughout the whole local training and peer-to-peer communication process. Comprehensive experiments demonstrate that DisPFL significantly saves the communication bottleneck for the busiest node among all clients and, at the same time, achieves higher model accuracy with less computation cost and communication rounds. Furthermore, we demonstrate that our method can easily adapt to heterogeneous local clients with varying computation complexities and achieves better personalized performances.


> Poster: Practical Almost-Linear-Time Approximation Algorithms for Hybrid and Overlapping Graph Clustering
> 
> Authors: Lorenzo Orecchia and Konstantinos Ameranis and Charalampos Tsourakakis and Kunal Talwar
> 
> Abstract: Detecting communities in real-world networks and clustering similarity graphs are major data mining tasks with a wide range of applications in graph mining, collaborative filtering, and bioinformatics. In many such applications, overwhelming empirical evidence suggests that communities and clusters are naturally overlapping, i.e., the boundary of a cluster may contain both edges across clusters  and  nodes  that  are  shared  with  other clusters, calling for novel hybrid graph partitioning algorithms (HGP).  While almost-linear-time approximation algorithms are known for edge-boundary-based graph partitioning, little progress has been made on fast algorithms for HGP, even in the special case of vertex-boundary-based graph partitioning. In this work, we introduce a frame-work based on two novel clustering objectives, which naturally extend the well-studied notion of  conductance  to  clusters  with  hybrid  vertex-and  edge-boundary  structure.   Our  main  algorithmic  contributions  are  almost-linear-time  algorithms O(log n)-approximation algorithms for both these objectives. To this end, we show that the cut-matching framework of (Khandekar et al., 2014) can be significantly extended to incorporate hybrid partitions.   Crucially,  we implement our approximation algorithm to produce both hybrid partitions and optimality certificates for large graphs, easily scaling to tens of millions of edges, and test our implementation on real-world datasets against other competitive baselines.


> Poster: Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning
> 
> Authors: Momin Abbas and Quan Xiao and Lisha Chen and Pin-Yu Chen and Tianyi Chen
> 
> Abstract: Model-agnostic meta learning (MAML) is currently one of the dominating approaches for few-shot meta-learning. Albeit its effectiveness, the training of MAML can be challenging due to the innate bilevel problem structure. Specifically, the loss landscape of MAML is much complex with possibly many more saddle points and local minima than its empirical risk minimization counterpart. To address this challenge, we leverage the recently invented sharpness-aware minimization and develop a sharpness-aware MAML approach that we term Sharp-MAML. We empirically demonstrate that Sharp-MAML and its computation-efficient variant can outperform popular existing MAML baselines (e.g., +12% accuracy on Mini-Imagenet). We complement the empirical study with the  convergence analysis and the generalization bound of Sharp-MAML. To the best of our knowledge, this is the first empirical and theoretical study on sharpness-aware minimization in the context of bilevel optimization. 


> Poster: Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer
> 
> Authors: Lucas N. Alegre and Ana Lucia Cetertich Bazzan and Bruno C. da Silva
> 
> Abstract: In many real-world applications, reinforcement learning (RL) agents might have to solve multiple tasks, each one typically modeled via a reward function. If reward functions are expressed linearly, and the agent has previously learned a set of policies for different tasks, successor features (SFs) can be exploited to combine such policies and identify reasonable solutions for new problems. However, the identified solutions are not guaranteed to be optimal. We introduce a novel algorithm that addresses this limitation. It allows RL agents to combine existing policies and directly identify optimal policies for arbitrary new problems, without requiring any further interactions with the environment. We first show (under mild assumptions) that the transfer learning problem tackled by SFs is equivalent to the problem of learning to optimize multiple objectives in RL. We then introduce an SF-based extension of the Optimistic Linear Support algorithm to learn a set of policies whose SFs form a convex coverage set. We prove that policies in this set can be combined via generalized policy improvement to construct optimal behaviors for any new linearly-expressible tasks, without requiring any additional training samples. We empirically show that our method outperforms state-of-the-art competing algorithms both in discrete and continuous domains under value function approximation.


> Poster: Efficient Approximate Inference for Stationary Kernel on Frequency domain
> 
> Authors: Yohan Jung and Kyungwoo Song and Jinkyoo Park
> 
> Abstract: Based on the Fourier duality between a stationary kernel and its spectral density, modeling the spectral density using a Gaussian mixture density enables one to construct a flexible kernel, known as a Spectral mixture kernel, that can model any stationary kernel. However, despite its expressive power, training this kernel is typically difficult because scalability and overfitting issues often arise due to a large number of training parameters. To resolve these issues, we propose an approximate inference method for estimating the Spectral mixture kernel hyperparameters. Specifically, we approximate this kernel by using the finite random spectral points based on Random Fourier Feature and optimize the parameters for the distribution of spectral points by applying sampling-based variational inference. To improve this inference procedure, we analyze the training loss and propose two special methods: a sampling method of spectral points to reduce the error of the approximate kernel in training, and an approximate natural gradient to accelerate the convergence of the parameter inference. 


> Poster: Correlation Clustering via Strong Triadic Closure Labeling: Fast Approximation Algorithms and Practical Lower Bounds
> 
> Authors: Nate Veldt
> 
> Abstract: Correlation clustering is a widely studied framework for clustering based on pairwise similarity and dissimilarity scores, but its best approximation algorithms rely on impractical convex relaxations. We present faster approximation algorithms that avoid these relaxations, by drawing new connections to edge labeling problems related to the principle of strong triadic closure. This includes faster and more practical linear programming algorithms, as well as extremely scalable combinatorial techniques, including the first combinatorial approximation algorithm for a variant of correlation clustering called cluster deletion. In practice, our algorithms produce approximate solutions that nearly match those of the canonical relaxation algorithms in quality, while scaling to graphs that are orders of magnitude larger.


> Poster: Random Forest Density Estimation
> 
> Authors: Hongwei Wen and Hanyuan Hang
> 
> Abstract: We propose a density estimation algorithm called \textit{random forest density estimation} (\textit{RFDE}) based on random trees where the split of cell is along the midpoint of the randomly chosen dimension. By combining the efficient random tree density estimation (RTDE) and the ensemble procedure, RFDE can alleviate the problems of boundary discontinuity suffered by partition-based density estimations. From the theoretical perspective, we first prove the fast convergence rates of RFDE if the density function lies in the H\"{o}lder space $C^{0,\alpha}$. Moreover, if the target function resides in the subspace $C^{1,\alpha}$, which contains smoother density functions, we for the first time manage to explain the benefits of the ensemble learning in density estimation. To be specific, we show that the upper bound of the ensemble estimator RFDE turns out to be strictly smaller than the lower bound of its base estimator RTDE in terms of convergence rates. In the experiments, we verify the theoretical results and show the promising performance of RFDE on both synthetic and real world datasets. Moreover, we evaluate our RFDE through the problem of anomaly detection as a possible application.

> Poster: PAC-Net: A Model Pruning Approach to Inductive Transfer Learning
> 
> Authors: Sanghoon Myung and In Huh and Wonik Jang and Jae Myung Choe and jisu ryu and Changwook Jeong and Daesin Kim and Kee-Eung Kim
> 
> Abstract: Inductive transfer learning aims to learn from a small amount of training data for the target task by utilizing a pre-trained model from the source task. Most strategies that involve large-scale deep learning models adopt initialization with the pre-trained model and fine-tuning for the target task. However, when using over-parameterized models, we can often prune the model without sacrificing the accuracy of the source task. This motivates us to adopt model pruning for transfer learning with deep learning models. In this paper, we propose PAC-Net, a simple yet effective approach for transfer learning based on pruning. PAC-Net consists of three steps: Prune, Allocate, and Calibrate (PAC). The main idea behind these steps is to identify essential weights for the source task, fine-tune on the source task by updating the essential weights, and then calibrate on the target task by updating the remaining redundant weights. Under the various and extensive set of inductive transfer learning experiments, we show that our method achieves state-of-the-art performance by a large margin.


> Poster: Uncertainty Modeling in Generative Compressed Sensing
> 
> Authors: Yilang Zhang and Mengchu Xu and Xiaojun Mao and Jian Wang
> 
> Abstract: Compressed sensing (CS) aims to recover a high-dimensional signal with  structural priors from its low-dimensional linear measurements. Inspired by the huge success of deep neural networks in modeling the priors of natural signals, generative neural networks have been recently used to replace the hand-crafted structural priors in CS. However, the reconstruction capability of the generative model is fundamentally limited by the range of its generator, typically a small subset of the signal space of interest. To break this bottleneck and thus reconstruct those out-of-range signals, this paper presents a novel method called CS-BGM that can effectively expands the range of generator. The key idea of CS-BGM is to introduce uncertainties to the latent variable and parameters of the generator,  while adopting the variational inference and maximum a posterior to infer them. Theoretical analysis shows that expanding the range of generators is necessary for the generative CS to reduce the reconstruction error. Extensive experiments demonstrate a consistent improvement of CS-BGM over baselines. 


> Poster: How to Leverage Unlabeled Data in Offline Reinforcement Learning?
> 
> Authors: Tianhe (Kevin) Yu and Aviral Kumar and Yevgen Chebotar and Karol Hausman and Chelsea Finn and Sergey Levine
> 
> Abstract: Offline reinforcement learning (RL) can learn control policies from static datasets but, like standard RL methods, it requires reward annotations for every transition. In many cases, labeling large datasets with rewards may be costly, especially if those rewards must be provided by human labelers, while collecting diverse unlabeled data might be comparatively inexpensive. How can we best leverage such unlabeled data in offline RL? One natural solution is to learn a reward function from the labeled data and use it to label the unlabeled data. In this paper, we find that, perhaps surprisingly, a much simpler method that simply applies zero rewards to unlabeled data leads to effective data sharing both in theory and in practice, without learning any reward model at all. While this approach might seem strange (and incorrect) at first, we provide extensive theoretical and empirical analysis that illustrates how it trades off reward bias, sample complexity and distributional shift, often leading to good results. We characterize conditions under which this simple strategy is effective, and further show that extending it with a simple reweighting approach can further alleviate the bias introduced by using incorrect reward labels. Our empirical evaluation confirms these findings in simulated robotic locomotion, navigation, and manipulation settings.


> Poster: PDE-Based Optimal Strategy for Unconstrained Online Learning
> 
> Authors: Zhiyu Zhang and Ashok Cutkosky and Ioannis Paschalidis
> 
> Abstract: Unconstrained Online Linear Optimization (OLO) is a practical problem setting to study the training of machine learning models. Existing works proposed a number of potential-based algorithms, but in general the design of such potential functions is ad hoc and heavily relies on guessing. In this paper, we present a framework that generates new potential functions by solving a Partial Differential Equation (PDE). As a concrete example, when losses are 1-Lipschitz, our framework produces a novel algorithm with anytime regret upper bound $C\sqrt{T}+||u||\sqrt{2T}[\sqrt{\log(1+||u||/C)}+2]$, where $C$ is a user-specified constant and $u$ is any comparator whose norm is unknown and unbounded a priori. By constructing a matching lower bound, we further show that the leading order term, including the constant multiplier $\sqrt{2}$, is tight. To our knowledge, this is the first parameter-free algorithm with the optimal leading constant. 

> Poster: Private frequency estimation via projective geometry
> 
> Authors: Vitaly Feldman and Jelani Nelson and Huy Nguyen and Kunal Talwar
> 
> Abstract: In  this work, we propose a new algorithm ProjectiveGeometryResponse (PGR) for locally differentially private (LDP) frequency estimation. For universe size of k and with n users, our eps-LDP algorithm has communication cost ceil(log_2 k) and computation cost O(n + k\exp(eps) log k) for the server to approximately reconstruct the frequency histogram, while achieving the state-of-the-art privacy-utility tradeoff. In many practical settings this is a significant improvement over the O~(n+k^2) computation cost that is achieved by the recent PI-RAPPOR algorithm (Feldman and Talwar; 2021). Our empirical evaluation shows a speedup of over 50x over PI-RAPPOR while using approximately 75x less memory. In addition, the running time of our algorithm is comparable to that of HadamardResponse (Acharya, Sun, and Zhang; 2019) and RecursiveHadamardResponse (Chen, Kairouz, and Ozgur; 2020) which have significantly worse reconstruction error. The error of our algorithm essentially matches that of the communication- and time-inefficient but utility-optimal SubsetSelection (SS) algorithm (Ye and Barg; 2017).  Our new algorithm is based on using Projective Planes over a finite field to define a small collection of sets that are close to being pairwise independent and a dynamic programming algorithm for approximate histogram reconstruction for the server.


> Poster: Deconfounded Value Decomposition for Multi-Agent Reinforcement Learning
> 
> Authors: Jiahui Li and Kun Kuang and Baoxiang Wang and Furui Liu and Long Chen and Changjie Fan and Fei Wu and Jun Xiao
> 
> Abstract: Value decomposition (VD) methods have been widely used in cooperative multi-agent reinforcement learning (MARL), where credit assignment plays an important role in guiding the agents’ decentralized execution.In this paper, we investigate VD from a novel perspective of causal inference. We first show that the environment in existing VD methods is an unobserved confounder as the common cause factor of the global state and the joint value function, which leads to the confounding bias on learning credit assignment.We then present our approach, deconfounded value decomposition (DVD), which cuts off the backdoor confounding path from the global state to the joint value function.The cut is implemented by introducing the trajectory graph, which depends only on the local trajectories, as a proxy confounder.DVD is general enough to be applied to various VD methods, and extensive experiments show that DVD can consistently achieve significant performance gains over different state-of-the-art VD methods on StarCraft II and MACO benchmarks.


> Poster: Counterfactual Prediction for Outcome-oriented Treatments
> 
> Authors: Hao Zou and Peng Cui and Bo Li and Jiangang Han and Shuiping Chen and Xuetao Ding
> 
> Abstract: Large amounts of efforts have been devoted into learning counterfactual treatment outcome under various settings, including binary/continuous/multiple treatments. Most of these literature aims to minimize the estimation error of counterfactual outcome for the whole treatment space. However, in most scenarios when the counterfactual prediction model is utilized to assist decision-making, people are only concerned with the small fraction of treatments that can potentially induce superior outcome (i.e. outcome-oriented treatments). This gap of objective is even more severe when the number of possible treatments is large, for example under the continuous treatment setting. To overcome it, we establish a new objective of optimizing counterfactual prediction on outcome-oriented treatments, propose a novel Outcome-oriented Sample Re-weighting(OOSR) method to make the predictive model concentrate more on outcome-oriented treatments, and theoretically analyze that our method can improve treatment selection towards the optimal one. Extensive experimental results on both synthetic datasets and semi-synthetic datasets demonstrate the effectiveness of our method.


> Poster: Markov Chain Monte Carlo for Continuous-Time Switching Dynamical Systems
> 
> Authors: Lukas Köhs and Bastian Alt and Heinz Koeppl
> 
> Abstract: Switching dynamical systems are an expressive model class for the analysis of time-series data. As in many fields within the natural and engineering sciences, the systems under study typically evolve continuously in time, it is natural to consider continuous-time model formulations consisting of switching stochastic differential equations governed by an underlying Markov jump process. Inference in these types of models is however notoriously difficult, and tractable computational schemes are rare. In this work, we propose a novel inference algorithm utilizing a Markov Chain Monte Carlo approach. The presented Gibbs sampler allows to efficiently obtain samples from the exact continuous-time posterior processes. Our framework naturally enables Bayesian parameter estimation, and we also include an estimate for the diffusion covariance, which is oftentimes assumed fixed in stochastic differential equations models. We evaluate our framework under the modeling assumption and compare it against an existing variational inference approach.


> Poster: Probabilistic ODE Solutions in Millions of Dimensions
> 
> Authors: Nicholas Krämer and Nathanael Bosch and Jonathan Schmidt and Philipp Hennig
> 
> Abstract: Probabilistic solvers for ordinary differential equations (ODEs) have emerged as an efficient framework for uncertainty quantification and inference on dynamical systems. In this work, we explain the mathematical assumptions and detailed implementation schemes behind solving high-dimensional ODEs with a probabilistic numerical algorithm. This has not been possible before due to matrix-matrix operations in each solver step, but is crucial for scientifically relevant problems---most importantly, the solution of discretised partial differential equations. In a nutshell, efficient high-dimensional probabilistic ODE solutions build either on independence assumptions or on Kronecker structure in the prior model. We evaluate the resulting efficiency on a range of problems, including the probabilistic numerical simulation of a differential equation with millions of dimensions.


> Poster: SPECTRE : Spectral Conditioning Overcomes the Expressivity Limits of One-shot Graph Generators
> 
> Authors: Karolis Martinkus and Andreas Loukas and Nathanaël Perraudin and Roger Wattenhofer
> 
> Abstract: We approach the graph generation problem from a spectral perspective by first generating the dominant parts of the graph Laplacian spectrum and then building a graph matching these eigenvalues and eigenvectors. Spectral conditioning allows for direct modeling of the global and local graph structure and helps to overcome the expressivity and mode collapse issues of one-shot graph generators.Our novel GAN, called SPECTRE, enables the one-shot generation of much larger graphs than previously possible with one-shot models. SPECTRE is the first one-shot model that outperforms state-of-the-art deep autoregressive generators in terms of modeling fidelity, while also avoiding expensive sequential generation and dependence on node ordering. A case in point, in large synthetic graphs SPECTRE achieves a 4-to-70 fold improvement over the best competitor that does not memorize the training set and is 17-to-34 times faster than autoregressive generators.


> Poster: When Are Linear Stochastic Bandits Attackable?
> 
> Authors: Huazheng Wang and Haifeng Xu and Hongning Wang
> 
> Abstract: We study adversarial attacks on linear stochastic bandits: by manipulating the rewards, an adversary aims to control the behaviour of the bandit algorithm. Perhaps surprisingly, we first show that some attack goals can never be achieved. This is in a sharp contrast to  context-free stochastic bandits, and is intrinsically due to the correlation among arms in linear stochastic bandits. Motivated by this finding, this paper studies the attackability of a k-armed linear bandit environment. We first provide a complete necessity and sufficiency characterization of attackability based on the geometry of the context vectors. We then propose a two-stage attack method against LinUCB and Robust Phase Elimination. The method first asserts whether the given environment is attackable; and if yes, it modifies the rewards to force the algorithm to pull a target arm linear times using only a sublinear cost. Numerical experiments further validate the effectiveness and cost-efficiency of the proposed attacking method. 


> Poster: Surrogate Likelihoods for Variational Annealed Importance Sampling
> 
> Authors: Martin Jankowiak and Du Phan
> 
> Abstract: Variational inference is a powerful paradigm for approximate Bayesian inference with a number of appealing properties, including support for model learning and data subsampling. By contrast MCMC methods like Hamiltonian Monte Carlo do not share these properties but remain attractive since, contrary to parametric methods, MCMC is asymptotically unbiased. For these reasons researchers have sought to combine the strengths of both classes of algorithms, with recent approaches coming closer to realizing this vision in practice. However, supporting data subsampling in these hybrid methods can be a challenge, a shortcoming that we address by introducing a surrogate likelihood that can be learned jointly with other variational parameters. We argue theoretically that the resulting algorithm allows an intuitive trade-off between inference fidelity and computational cost. In an extensive empirical comparison we show that our method performs well in practice and that it is well-suited for black-box inference in probabilistic programming frameworks.


> Poster: Understanding Gradient Descent on the Edge of Stability in Deep Learning
> 
> Authors: Sanjeev Arora and Zhiyuan Li and Abhishek Panigrahi
> 
> Abstract: Deep learning experiments in ~\cite{cohen2021gradient} using deterministic Gradient Descent (GD) revealed an {\em Edge of Stability (EoS)} phase when learning rate (LR) and sharpness (\emph{i.e.}, the largest eigenvalue of hessian) no longer behave as in traditional optimization. Sharpness stabilizes around $2/$LR and loss goes up and down across iterations, yet still with an overall downward trend. The current paper mathematically analyzes a new mechanism of implicit regularization in the EoS phase, whereby GD updates due to non-smooth loss landscape turn out to evolve along some deterministic flow on the manifold of minimum loss. This is in contrast to many previous results about implicit bias either relying on infinitesimal updates or noise in gradient. Formally, for any smooth function $L$ with certain regularity condition, this effect is demonstrated for (1) {\em Normalized GD}, i.e., GD  with a varying LR $\eta_t = \frac{\eta}{\norm{\nabla L(x(t))}}$ and loss $L$;  (2) GD with constant LR and loss $\sqrt{L}$.  Both provably enter the Edge of Stability, with the associated flow on the manifold minimizing $\lambda_{\max}(\nabla^2 L)$. The above theoretical results have been corroborated by an experimental study.

> Poster: Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense
> 
> Authors: Bao Gia Doan and Ehsan Abbasnejad and Javen Qinfeng Shi and Damith Ranashinghe
> 
> Abstract: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the learning approach for approximating the multi-modal posterior distribution of an adversarially trained Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore  perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step towards a basis for a principled method of adversarially training BNNs. Our extensive experimental results demonstrate significantly improved robustness up to 20% compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 dataset.


> Poster: Fast Relative Entropy Coding with A* coding
> 
> Authors: Gergely Flamich and Stratis Markou and Jose Miguel Hernandez-Lobato
> 
> Abstract: Relative entropy coding (REC) algorithms encode a sample from a target distribution Q using a proposal distribution P, such that the expected codelength is O(KL[Q || P]). REC can be seamlessly integrated with existing learned compression models since, unlike entropy coding, it does not assume discrete Q or P, and does not require quantisation. However, general REC algorithms require an intractable  Ω(exp(KL[Q || P])) runtime. We introduce AS* and AD* coding, two REC algorithms based on A* sampling. We prove that, for continuous distributions over the reals, if the density ratio is unimodal, AS* has O(D∞[Q || P]) expected runtime, where D∞[Q || P] is the Renyi ∞-divergence. We provide experimental evidence that AD* also has O(D∞[Q || P]) expected runtime. We prove that AS* and AD* achieve an expected codelength of O(KL[Q || P]). Further, we introduce DAD, an approximate algorithm based on AD which retains its favourable runtime and has bias similar to that of alternative methods. Focusing on VAEs, we propose the IsoKL VAE (IKVAE), which can be used with DAD* to further improve compression efficiency. We evaluate A* coding with (IK)VAEs on MNIST, showing that it can losslessly compress images near the theoretically optimal limit.


> Poster: Cross-Space Active Learning on Graph Convolutional Networks
> 
> Authors: Yufei Tao and Hao WU and Shiyuan Deng
> 
> Abstract: This paper formalizes {\em cross-space} active vertex classification on a graph convolutional network (GCN). The objective is to attain the best accuracy that can be achieved in some feature space among the multiple feature spaces generated by the GCN. Subjective to the objective, the challenge is to minimize the {\em label cost}, measured in the number of vertices that should be labeled. Matching upper and lower bounds are established for the label complexities of two algorithm classes that differ in whether a special clue about the best feature space is accepted. The clue is vital and promises strictly a lower complexity in numerous scenarios. 


> Poster: Transfer and Marginalize: Explaining Away Label Noise with Privileged Information
> 
> Authors: Mark Collier and Rodolphe Jenatton and Effrosyni Kokiopoulou and Jesse Berent
> 
> Abstract: Supervised learning datasets often have privileged information, in the form of features which are available at training time but are not available at test time e.g. the ID of the annotator that provided the label. We argue that privileged information is useful for explaining away label noise, thereby reducing the harmful impact of noisy labels. We develop a simple and efficient method for supervised learning with neural networks: it transfers via weight sharing the knowledge learned with privileged information and approximately marginalizes over privileged information at test time. Our method, TRAM (TRansfer and Marginalize), has minimal training time overhead and has the same test-time cost as not using privileged information. TRAM performs strongly on CIFAR-10H, ImageNet and Civil Comments benchmarks.


> Poster: Modeling Irregular Time Series with Continuous Recurrent Units
> 
> Authors: Mona Schirmer and Mazin Eltayeb and Stefan Lessmann and Maja Rudolph
> 
> Abstract: Recurrent neural networks (RNNs) with a gating mechanism, such as long short-term memory networks (LSTMs) are a popular choice for modeling sequential data. Their gating mechanism permits weighting previous history encoded in a hidden state with new information from incoming observations. Standard RNNs assume constant time intervals between observations. However, in many applications such as health care, observation times are irregular and can carry important information. To address this challenge, we propose continuous recurrent units (CRUs) - a neural architecture that can naturally handle irregular time intervals between observations. The gating mechanism of the CRU employs a continuous formulation of the Kalman filter and alternates between (1) continuous latent state propagation according to a linear stochastic differential equation (SDE)  and (2) latent state updates whenever a new observation comes in. In an empirical study, we show that the CRU can better interpolate irregular time series than neural ordinary differential equation (neural ODE)-based models. We also show that our model can infer dynamics from images and that the Kalman gain efficiently singles out candidates for valuable state updates from noisy observations.


> Poster: Regularizing a Model-based Policy Stationary Distribution to Stabilize Offline Reinforcement Learning
> 
> Authors: shentao yang and Yihao Feng and Shujian Zhang and Mingyuan Zhou
> 
> Abstract: Offline reinforcement learning (RL) extends the paradigm of classical RL algorithms to purely learning from static datasets, without interacting with the underlying environment during the learning process. A key challenge of offline RL is the instability of policy training, caused by the mismatching between the distribution of the offline data and the undiscounted stationary state-action distribution of the learned policy. To avoid the detrimental impact of distribution mismatching, we regularize the undiscounted stationary distribution of the current policy towards the offline data during the policy improvement. Further, we train a dynamics model for regularization and to better estimate the stationary distribution of the current policy, reducing the error induced by distribution mismatch. On a wide range of continuous-control offline datasets, our method indicates competitive performance, which validates our algorithm.


> Poster: Conformal Prediction Sets with Limited False Positives
> 
> Authors: Adam Fisch and Tal Schuster and Tommi Jaakkola and Regina Barzilay
> 
> Abstract: We develop a new approach to multi-label conformal prediction in which we aim to output a precise set of promising prediction candidates with a bounded number of incorrect answers. Standard conformal prediction provides the ability to adapt to model uncertainty by constructing a calibrated candidate set in place of a single prediction, with guarantees that the set contains the correct answer with high probability. In order to obey this coverage property, however, conformal sets can become inundated with noisy candidates---which can render them unhelpful in practice. This is particularly relevant to practical applications where there is a limited budget, and the cost (monetary or otherwise) associated with false positives is non-negligible. We propose to trade coverage for a notion of precision by enforcing that the presence of incorrect candidates in the predicted conformal sets (i.e., the total number of false positives) is bounded according to a user-specified tolerance. Subject to this constraint, our algorithm then optimizes for a generalized notion of set coverage (i.e., the true positive rate) that allows for any number of true answers for a given query (including zero). We demonstrate the effectiveness of this approach across a number of classification tasks in natural language processing, computer vision, and computational chemistry.


> Poster: Robust SDE-based variational formulations for solving linear PDEs via deep learning
> 
> Authors: Lorenz Richter and Julius Berner
> 
> Abstract: The combination of Monte Carlo methods and deep learning has recently led to efficient algorithms for solving partial differential equations (PDEs) in high dimensions. Related learning problems are often stated as variational formulations based on associated stochastic differential equations (SDEs), which allow the minimization of corresponding losses using gradient-based optimization methods. In respective numerical implementations it is therefore crucial to rely on adequate gradient estimators that exhibit low variance in order to reach convergence accurately and swiftly. In this article, we rigorously investigate corresponding numerical aspects that appear in the context of linear Kolmogorov PDEs. In particular, we systematically compare existing deep learning approaches and provide theoretical explanations for their performances. Subsequently, we suggest novel methods that can be shown to be more robust both theoretically and numerically, leading to substantial performance improvements.


> Poster: Guaranteed Robust Deep Learning against Extreme Label Noise using Self-supervised Learning
> 
> Authors: Yihao Xue and Kyle Whitecross and Baharan Mirzasoleiman
> 
> Abstract: Self-supervised contrastive learning has been recently shown very effective in preventing deep networks from overfitting noisy labels. Despite its empirical success, the theoretical understanding of the effect of contrastive learning on boosting robustness of deep networks is very limited. In this work, we show that contrastive learning provably boosts robustness of deep networks against noisy labels by providing an embedding matrix that has (i) a singular value corresponding to each subclass in the data, which is relatively larger than the sum of the remaining singular values of that subclass; and (ii) a large alignment between the largest singular vector and the clean labels of that subclass. The above properties allow a linear layer trained on the embeddings to learn the clean labels quickly, and prevent it from overfitting the noisy labels for a large number of training iterations.We further show that the initial robustness provided by contrastive learning enables state-of-the-art robust methods to achieve a superior performance under extreme noise levels, e.g., 6.3\% increase in accuracy on CIFAR-10 with 40\% asymmetric noisy labels, and 14\% increase in accuracy on CIFAR100 with 80\% symmetric noisy labels.


> Poster: Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation
> 
> Authors: Volodymyr Kuleshov and Shachi Deshpande
> 
> Abstract: Predictive uncertainties can be characterized by two properties---calibration and sharpness. This paper argues for reasoning about uncertainty in terms of these properties and proposes simple algorithms for enforcing them in predictive models. % deep learning. Our methods focus on the strongest notion of calibration---distribution calibration---and show that enforcing it is as simple as performing low-dimensional density estimation. The resulting approach is much simpler and more broadly applicable than previous methods across both classification and regression. They are also simpler and more broadly applicable than previous methods. Moreover, it implements a long-standing principle that forecasts should ``maximize sharpness subject to being calibrated". This lends support for reasoning about uncertainty using calibration and sharpness and suggest simple and improved ways of training deep learning models that should be leveraged to improve performance across downstream applications.


> Poster: On the Equivalence Between Temporal and Static Equivariant Graph Representations
> 
> Authors: Jianfei Gao and Bruno Ribeiro
> 
> Abstract: This work formalizes the associational task of predicting node attribute evolution in temporal graphs from the perspective of learning equivariant representations. We show that node representations in temporal graphs can be cast into two distinct frameworks: (a) The most popular approach, which we denote as time-and-graph, where equivariant graph (e.g., GNN) and sequence (e.g., RNN) representations are intertwined to represent the temporal evolution of node attributes in the graph; and (b) an approach that we denote as time-then-graph, where the sequences describing the node and edge dynamics are represented first, then fed as node and edge attributes into a static equivariant graph representation that comes after. Interestingly, we show that time-then-graph representations have an expressivity advantage over time-and-graph representations when both use component GNNs that are not most-expressive (e.g., 1-Weisfeiler-Lehman GNNs). Moreover, while our goal is not necessarily to obtain state-of-the-art results, our experiments show that time-then-graph methods are capable of achieving better performance and efficiency than state-of-the-art time-and-graph methods in some real-world tasks, thereby showcasing that the time-then-graph framework is a worthy addition to the graph ML toolbox.


> Poster: Learning Symmetric Embeddings for Equivariant World Models
> 
> Authors: Jung Yeon Park and Ondrej Biza and Linfeng Zhao and Jan-Willem van de Meent and Robin Walters
> 
> Abstract: Incorporating symmetries can lead to highly data-efficient and generalizable models by defining equivalence classes of data samples related by transformations. However, characterizing how transformations act on input data is often difficult, limiting the applicability of equivariant models. We propose learning symmetric embedding networks (SENs) that encode an input space (e.g. images), where we do not know the effect of transformations (e.g. rotations), to a feature space that transforms in a known manner under these operations. This network can be trained end-to-end with an equivariant task network to learn an explicitly symmetric representation. We validate this approach in the context of equivariant transition models with 3 distinct forms of symmetry. Our experiments demonstrate that SENs facilitate the application of equivariant networks to data with complex symmetry representations. Moreover, doing so can yield improvements in accuracy and generalization relative to both fully-equivariant and non-equivariant baselines.


> Poster: Towards Coherent and Consistent Use of Entities in Narrative Generation
> 
> Authors: Pinelopi Papalampidi and Kris Cao and Tomas Kocisky
> 
> Abstract: Large pre-trained language models (LMs) have demonstrated impressive capabilities in generating long, fluent text; however, there is little to no analysis on their ability to maintain entity coherence and consistency. In this work, we focus on the end task of narrative generation and systematically analyse the long-range entity coherence and consistency in generated stories. First, we propose a set of automatic metrics for measuring model performance in terms of entity usage. Given these metrics, we quantify the limitations of current LMs. Next, we propose augmenting a pre-trained LM with a dynamic entity memory in an end-to-end manner by using an auxiliary entity-related loss for guiding the reads and writes to the memory. We demonstrate that the dynamic entity memory increases entity coherence according to both automatic and human judgment and helps preserving entity-related information especially in settings with a limited context window. Finally, we also validate that our automatic metrics are correlated with human ratings and serve as a good indicator of the quality of generated stories.  


> Poster: Structured Stochastic Gradient MCMC
> 
> Authors: Antonios Alexos and Alex Boyd and Stephan Mandt
> 
> Abstract: Stochastic gradient Markov Chain Monte Carlo (SGMCMC) is considered the gold standard for Bayesian inference in large-scale models, such as Bayesian neural networks. Since practitioners face speed versus accuracy tradeoffs in these models, variational inference (VI) is often the preferable option. Unfortunately, VI makes strong assumptions on both the factorization and functional form of the posterior. In this work, we propose a new non-parametric variational approximation that makes no assumptions about the approximate posterior's functional form and allows practitioners to specify the exact dependencies the algorithm should respect or break. The approach relies on a new Langevin-type algorithm that operates on a modified energy function, where parts of the latent variables are averaged over samples from earlier iterations of the Markov chain. This way, statistical dependencies can be broken in a controlled way, allowing the chain to mix faster. This scheme can be further modified in a ``dropout'' manner, leading to even more scalability. We test our scheme for ResNet-20 on CIFAR-10, SVHN, and FMNIST. In all cases, we find improvements in convergence speed and/or final accuracy compared to SG-MCMC and VI.


> Poster: GSmooth: Certified Robustness against Semantic Transformations via Generalized Randomized Smoothing
> 
> Authors: Zhongkai Hao and Chengyang Ying and Yinpeng Dong and Hang Su and Jian Song and Jun Zhu
> 
> Abstract: Certified defenses such as randomized smoothing have shown promise towards building reliable machine learning systems against $\ell_p$ norm bounded attacks. However, existing methods are insufficient or unable to provably defend against semantic transformations, especially those without closed-form expressions (such as defocus blur and pixelate), which are more common in practice and often unrestricted.  To fill up this gap, we propose generalized randomized smoothing (GSmooth), a unified theoretical framework for certifying robustness against general semantic transformations via a novel dimension augmentation strategy. Under the GSmooth framework, we present a scalable algorithm that uses a surrogate image-to-image network to approximate the complex transformation. The surrogate model provides a powerful tool for studying the properties of semantic transformations and certifying robustness. Experimental results on several datasets demonstrate the effectiveness of our approach for robustness certification against multiple kinds of semantic transformations and corruptions, which is not achievable by the alternative baselines.

> Poster: Self-supervised learning with random-projection quantizer for speech recognition
> 
> Authors: Chung-Cheng Chiu and James Qin and Yu Zhang and Jiahui Yu and Yonghui Wu
> 
> Abstract: We present a simple and effective self-supervised learning approach for speech recognition. The approach learns a model to predict the masked speech signals, in the form of discrete labels generated with a random-projection quantizer. In particular the quantizer projects speech inputs with a randomly initialized matrix, and does a nearest-neighbor lookup in a randomly-initialized codebook. Neither the matrix nor the codebook are updated during self-supervised learning. Since the random-projection quantizer is not trained and is separated from the speech recognition model, the design makes the approach flexible and is compatible with universal speech recognition architecture. On LibriSpeech our approach achieves similar word-error-rates as previous work using self-supervised learning with non-streaming models, and provides lower word-error-rates than previous work with streaming models. On multilingual tasks the approach also provides significant improvement over wav2vec 2.0 and w2v-BERT.


> Poster: On the Statistical Benefits of Curriculum Learning
> 
> Authors: Ziping Xu and Ambuj Tewari
> 
> Abstract: Curriculum learning (CL) is a commonly used machine learning training strategy. However, we still lack a clear theoretical understanding of CL's benefits. In this paper, we study the benefits of CL in the multitask linear regression problem under both structured and unstructured settings. For both settings, we derive the minimax rates for CL with the oracle that provides the optimal curriculum and without the oracle, where the agent has to adaptively learn a good curriculum. Our results reveal that adaptive learning can be fundamentally harder than the oracle learning in the unstructured setting, but it merely introduces a small extra term in the structured setting. To connect theory with practice, we provide justification for a popular empirical method that selects tasks with highest local prediction gain by comparing its guarantees with the minimax rates mentioned above.


> Poster: Reinforcement Learning with Action-Free Pre-Training from Videos
> 
> Authors: Younggyo Seo and Kimin Lee and Stephen James and Pieter Abbeel
> 
> Abstract: Recent unsupervised pre-training methods have shown to be effective on language and vision domains by learning useful representations for multiple downstream tasks. In this paper, we investigate if such unsupervised pre-training methods can also be effective for vision-based reinforcement learning (RL). To this end, we introduce a framework that learns representations useful for understanding the dynamics via generative pre-training on videos. Our framework consists of two phases: we pre-train an action-free latent video prediction model, and then utilize the pre-trained representations for efficiently learning action-conditional world models on unseen environments. To incorporate additional action inputs during fine-tuning, we introduce a new architecture that stacks an action-conditional latent prediction model on top of the pre-trained action-free prediction model. Moreover, for better exploration, we propose a video-based intrinsic bonus that leverages pre-trained representations. We demonstrate that our framework significantly improves both final performances and sample-efficiency of vision-based RL in a variety of manipulation and locomotion tasks.


> Poster: Biased Gradient Estimate with Drastic Variance Reduction for Meta Reinforcement Learning
> 
> Authors: Yunhao Tang
> 
> Abstract: Despite the empirical success of meta reinforcement learning (meta-RL), there are still a number poorly-understood discrepancies between theory and practice. Critically, biased gradient estimates are almost always implemented in practice, whereas prior theory on meta-RL only establishes convergence under unbiased gradient estimates. In this work, we investigate such a discrepancy. In particular, (1) We show that unbiased gradient estimates have variance $\Theta(N)$ which linearly depends on the sample size $N$ of the inner loop updates; (2) We propose linearized score function (LSF) gradient estimates, which have bias $\mathcal{O}(1/\sqrt{N})$ and variance $\mathcal{O}(1/N)$; (3) We show that most empirical prior work in fact implements variants of the LSF gradient estimates. This implies that practical algorithms "accidentally" introduce bias to achieve better performance; (4) We establish theoretical guarantees for the LSF gradient estimates in meta-RL regarding its convergence to stationary points, showing better dependency on $N$ than prior work when $N$ is large.

> Poster: Provable Acceleration of Heavy Ball beyond Quadratics for a class of Polyak-Lojasiewicz Functions when the Non-Convexity is Averaged-Out
> 
> Authors: Jun-Kun Wang and Chi-Heng Lin and Andre Wibisono and Bin Hu
> 
> Abstract: Heavy Ball (HB) nowadays is one of the most popular momentum methods in non-convex optimization. It has been widely observed that incorporating the Heavy Ball dynamic in gradient-based methods accelerates the training process of modern machine learning models. However, the progress on establishing its theoretical foundation of acceleration is apparently far behind its empirical success. Existing provable acceleration results are of the quadratic or close-to-quadratic functions, as the current techniques of showing HB's acceleration are limited to the case when the Hessian is fixed. In this work, we develop some new techniques that help show acceleration beyond quadratics, which is achieved by analyzing how the change of the Hessian at two consecutive time points affects the convergence speed. Based on our technical results, a class of Polyak-Lojasiewicz (PL) optimization problems for which provable acceleration can be achieved via HB is identified. Moreover, our analysis demonstrates a benefit of adaptively setting the momentum parameter.


> Poster: MonePipe: Accelerating Momentum Network Training with Pipelines
> 
> Authors: Hwijoon Lim and Yechan Kim and Jinwoo Shin and Dongsu Han
> 
> Abstract: Self-supervised learning (SSL) has shown great success in learning visual representations from unlabeled images. A common training principle underlying recent SSL frameworks is to maintain an auxiliary network called momentum network, which does not require a backward pass and is slowly updated. SSL networks generally have notoriously large training costs, as they require larger architectures and longer training epochs to converge. In this paper, we present MonePipe, the pipelined approach to accelerate the training process of momentum networks. Under the observation that the target momentum network does not need a backward pass, our main idea is to fully utilize GPU during training by scheduling computation of the target network when the pipeline is idle. Furthermore, we suggest using delayed parameter updates only on the target network, for attaining high model accuracy. Compared to existing pipeline parallelism schemes, which sacrifice either training throughput or model accuracy, MonePipe provides better performance trade-offs. MonePipe achieves 6.42x higher training throughput without loss of the model accuracy compared to the inter-layer MP baseline when training a MoCo-v3 model.


> Poster: Faster Algorithms for Learning Convex Functions
> 
> Authors: Ali Siahkamari and Durmus Alp Emre Acar and Christopher Liao and Kelly Geyer and Venkatesh Saligrama and Brian Kulis
> 
> Abstract: The task of approximating an arbitrary convex function arises in several learning problems such as convex regression, learning with a difference of convex (DC) functions, and learning Bregman or $f$-divergences. In this paper, we develop and analyze an approach for solving a broad range of convex function learning problems that is faster than state-of-the-art approaches.  Our approach is based on a 2-block ADMM method where each block can be computed in closed form.  For the task of convex Lipschitz regression, we establish that our proposed algorithm converges with iteration complexity of $ O(n\sqrt{d}/\epsilon)$   for a dataset $ X \in  R^{n\times d}$ and $\epsilon > 0$. Combined with per-iteration computation complexity, our method converges with the rate $O(n^3 d^{1.5}/\epsilon+n^2 d^{2.5}/\epsilon+n d^3/\epsilon)$. This new rate improves the state of the art  rate of $O(n^5d^2/\epsilon)$ available by interior point methods if $d = o( n^4)$.  Further we provide similar solvers for DC regression and Bregman divergence learning.  Unlike previous approaches, our method is amenable to the use of GPUs.   We demonstrate on regression and metric learning experiments that our approach is over 100 times faster than existing approaches on some data sets, and produces results that are comparable to state of the art.

> Poster: Benefits of Deep and Wide Convolutional Residual Networks: Function Approximation under Smoothness Constraint
> 
> Authors: Hao Liu and Minshuo Chen and Siawpeng Er and Wenjing Liao and Tong Zhang and Tuo Zhao
> 
> Abstract: Large (deep and wide) neural networks enjoy great representation power on complex data, and more importantly yield sufficiently smooth output, which is crucial to their generalization and robustness. Existing function approximation theories only suggest that with sufficiently many parameters, neural networks can well approximate certain classes of functions in terms of the function value. The neural network themselves, however, can be highly nonsmooth. To bridge this gap, we take convolutional residual networks (ConvResNets) as an example, and prove that large ConvResNets can not only approximate a target function in terms of function value, but also exhibit sufficient first-order smoothness. Moreover, we extend our theory to approximating functions supported on a low-dimensional manifold. Our theory partially justifies the benefits of using deep and wide networks in practice. Numerical experiments on adversarial robust image classification are provided to support our theory.


> Poster: Balancing Sample Efficiency and Suboptimality in Inverse Reinforcement Learning
> 
> Authors: Giorgio Manganini and Angelo Damiani and Alberto Maria Metelli and Marcello Restelli
> 
> Abstract: We propose a novel formulation for the Inverse Reinforcement Learning (IRL) problem, which jointly accounts for the compatibility with the expert behavior of the identified reward and its effectiveness for the subsequent forward learning phase. Albeit quite natural, especially when the final goal is apprenticeship learning (learning policies from an expert), this aspect has been completely overlooked by IRL approaches so far.We propose a new model-free IRL method that is remarkably able to autonomously find a trade-off between the error induced on the learned policy when potentially choosing a sub-optimal reward, and the estimation error caused by using finite samples in the forward learning phase, which can be controlled by explicitly optimizing also the discount factor of the related learning problem. The approach is based on a min-max formulation for the robust selection of the reward parameters and the discount factor so that the distance between the expert's policy and the learned policy is minimized in the successive forward learning task when a finite and possibly small number of samples is available.Differently from the majority of other IRL techniques, our approach does not involve any planning or forward Reinforcement Learning problems to be solved. After presenting the formulation, we provide a numerical scheme for the optimization, and we show its effectiveness on an illustrative numerical case.


> Poster: Task-aware Privacy Preservation for Multi-dimensional Data
> 
> Authors: Jiangnan Cheng and Ao Tang and Sandeep Chinchali
> 
> Abstract: Local differential privacy (LDP) can be adopted to anonymize richer user data attributes that will be input to sophisticated machine learning (ML) tasks.  However, today's LDP approaches are largely task-agnostic and often lead to severe performance loss -- they simply inject noise to all data attributes according to a given privacy budget, regardless of what features are most relevant for the ultimate task. In this paper, we address how to significantly improve the ultimate task performance with multi-dimensional user data by considering a task-aware privacy preservation problem. The key idea is to use an encoder-decoder framework to learn (and anonymize) a task-relevant latent representation of user data.  We obtain an analytical near-optimal solution for the linear setting with mean-squared error (MSE) task loss. We also provide an approximate solution through a learning algorithm for general nonlinear cases. Extensive experiments demonstrate that our task-aware approach significantly improves ultimate task accuracy compared to standard benchmark LDP approaches with the same level of privacy guarantee.


> Poster: Retroformer: Pushing the Limits of End-to-end Retrosynthesis Transformer
> 
> Authors: Yue Wan and Chang-Yu (Kim) Hsieh and Shengyu Zhang and Ben Liao
> 
> Abstract: Retrosynthesis prediction is one of the fundamental challenges in organic synthesis. The task is to predict the reactants given a core product. With the advancement of machine learning, computer-aided synthesis planning has gained increasing interest. Numerous methods were proposed to solve this problem with different levels of dependency on additional chemical knowledge. In this paper, we propose Retroformer, a novel Transformer-based architecture for retrosynthesis prediction without relying on any cheminformatics tools for molecule editing. Via the proposed local attention head, the model can jointly encode the molecular sequence and graph, and efficiently exchange information between the local reactive region and the global reaction context. Retroformer reaches the new state-of-the-art accuracy for the end-to-end template-free retrosynthesis, and improves over many strong baselines on better molecule and reaction validity. In addition, its generative procedure is highly interpretable and controllable. Overall, Retroformer pushes the limits of the reaction reasoning ability of deep generative models.


> Poster: SMODICE: Versatile Offline Imitation Learning via State Occupancy Matching
> 
> Authors: Yecheng Ma and Andrew Shen and Dinesh Jayaraman and Osbert Bastani
> 
> Abstract: We propose State Matching Offline DIstribution Correction Estimation (SMODICE), a novel and versatile algorithm for offline imitation learning (IL) via state-occupancy matching. We show that the SMODICE objective admits a simple optimization procedure through an application of Fenchel duality and an analytic solution in tabular MDPs. Without requiring access to expert actions, SMODICE can be effectively applied to three offline IL settings: (i) imitation from observations (IfO), (ii) IfO with dynamics or morphologically mismatched expert, and (iii) example-based reinforcement learning, which we show can be formulated as a state-occupancy matching problem. We extensively evaluate SMODICE on both gridworld environments as well as on high-dimensional offline benchmarks. Our results demonstrate that SMODICE is effective for all three problem settings and significantly outperforms prior state-of-art. 


> Poster: Policy Gradient Method For Robust Reinforcement Learning
> 
> Authors: Yue Wang and Shaofeng Zou
> 
> Abstract: This paper develops the first policy gradient method with global optimality guarantee and complexity analysis for robust reinforcement learning under model mismatch. Robust reinforcement learning is to learn a policy robust to model mismatch between simulator and real environment. We first develop the robust policy (sub-)gradient, which is applicable for any differentiable parametric policy class. We show that the proposed robust policy gradient method converges to the global optimum asymptotically under direct policy parameterization. We further develop a smoothed robust policy gradient method, and show that to achieve an $\epsilon$-global optimum, the complexity is $\mathcal O(\epsilon^{-3})$. We then extend our methodology to the general model-free setting, and design the robust actor-critic method with differentiable parametric policy class and value function. We further characterize its asymptotic convergence and sample complexity under the tabular setting. Finally, we provide simulation results to demonstrate the robustness of our methods.

> Poster: Stability Based Generalization Bounds for Exponential Family Langevin Dynamics
> 
> Authors: Arindam Banerjee and Tiancong Chen and Xinyan Li and Yingxue Zhou
> 
> Abstract: Recent years have seen advances in generalization bounds for noisy stochastic algorithms, especially stochastic gradient Langevin dynamics (SGLD) based on stability (Mou et al., 2018; Li et al., 2020) and information theoretic approaches (Xu & Raginsky, 2017; Negrea et al., 2019; Steinke & Zakynthinou, 2020). In this paper, we unify and substantially generalize stability based generalization bounds and make three technical contributions. First, we bound the generalization error in terms of expected (not uniform) stability which arguably leads to quantitatively sharper bounds. Second, as our main contribution, we introduce Exponential Family Langevin Dynamics (EFLD), a substantial generalization of SGLD, which includes noisy versions of Sign-SGD and quantized SGD as special cases. We establish data dependent expected stability based generalization bounds for any EFLD algorithm with a O(1/n) sample dependence and dependence on gradient discrepancy rather than the norm of gradients, yielding significantly sharper bounds. Third, we establish optimization guarantees for special cases of EFLD. Further, empirical results on benchmarks illustrate that our bounds are non-vacuous, quantitatively sharper than existing bounds, and behave correctly under noisy labels.


> Poster: Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization
> 
> Authors: Samuel Hurault and Nicolas Papadakis and Arthur Leclaire
> 
> Abstract: Plug-and-Play (PnP) methods solve ill-posed inverse problems through iterative proximal algorithms by replacing a proximal operator by a denoising operation. When applied with deep neural network denoisers, these methods have shown state-of-the-art visual performance for image restoration problems. However, their theoretical convergence analysis is still incomplete. Most of the existing convergence results consider nonexpansive denoisers, which is non-realistic, or limit their analysis to strongly convex data-fidelity terms in the inverse problem to solve. Recently, it was proposed to train the denoiser as a gradient descent step on a functional parameterized by a deep neural network. Using such a denoiser guarantees the convergence of the PnP version of the Half-Quadratic-Splitting (PnP-HQS) iterative algorithm. In this paper, we show that this gradient denoiser can actually correspond to the proximal operator of another scalar function. Given this new result, we exploit the convergence theory of proximal algorithms in the nonconvex setting to obtain convergence results for PnP-PGD (Proximal Gradient Descent) and PnP-ADMM (Alternating Direction Method of Multipliers). When built on top of a smooth gradient denoiser, we show that PnP-PGD and PnP-ADMM are convergent and target stationary points of an explicit functional. These convergence results are confirmed with numerical experiments on deblurring, super-resolution and inpainting.


> Poster: Prototype-anchored Learning for Learning with Imperfect Annotations
> 
> Authors: Xiong Zhou and Xianming Liu and Deming Zhai and Junjun Jiang and Xin Gao and Xiangyang Ji
> 
> Abstract: The success of deep neural networks greatly relies on the availability of large amounts of high-quality annotated data, which however are difficult or expensive to obtain. The resulting labels may be class imbalanced, noisy or human biased. It is challenging to learn unbiased classification models from imperfectly annotated datasets, which usually suffer from overfitting or underfitting. In this work, we thoroughly investigate the popular softmax loss and margin-based loss, and offer a feasible approach to tighten the generalization error bound by maximizing the minimal sample margin. We further derive the optimality condition for this purpose, which indicates how the class prototypes should be anchored. Motivated by theoretical analysis, we propose a simple yet effective method, namely prototype-anchored learning (PAL), which can be easily incorporated into various learning-based classification schemes to handle imperfect annotation. We verify the effectiveness of PAL on class-imbalanced learning and noise-tolerant learning by extensive experiments on synthetic and real-world datasets.


> Poster: Towards Uniformly Superhuman Autonomy via Subdominance Minimization
> 
> Authors: Brian Ziebart and Sanjiban Choudhury and Xinyan Yan and Paul Vernaza
> 
> Abstract: Prevalent imitation learning methods seek to match average human performance by learning cost functions that induce similar behavior.  We instead assume demonstrations are of varying quality and seek to exceed human performance by inducing unambiguously better behavior (i.e., Pareto dominant or minimally subdominant). Our training objective is primarily defined by higher quality demonstrations; lower quality demonstrations, which are more easily dominated, are effectively ignored instead of degrading imitation. Our imitation learner, with increasing probability, produces superhuman behavior that incurs lower cost than the demonstrator on the demonstrator’s unknown cost function—even if that cost function differs for each demonstration. We apply our developed algorithms on a computer cursor pointing task, producing behavior that is 78% superhuman, while methods seeking to make demonstrations near optimal are 50% superhuman


> Poster: Analysis of Stochastic Processes through Replay Buffers
> 
> Authors: Shirli Di-Castro Shashua and Shie Mannor and Dotan Di Castro
> 
> Abstract: Replay buffers are a key component in many reinforcement learning schemes. Yet, their theoretical properties are not fully understood. In this paper we analyze a system where a stochastic process X is pushed into a replay buffer and then randomly sampled to generate a stochastic process Y from the replay buffer. We provide an analysis of the properties of the sampled process such as stationarity, Markovity and autocorrelation in terms of the properties of the original process. Our theoretical analysis sheds light on why replay buffer may be a good de-correlator. Our analysis provides theoretical tools for proving the convergence of replay buffer based algorithms which are prevalent in reinforcement learning schemes.


> Poster: Strategic Instrumental Variable Regression: Recovering Causal Relationships From Strategic Responses
> 
> Authors: Keegan Harris and Dung Ngo and Logan Stapleton and Hoda Heidari and Steven Wu
> 
> Abstract: In settings where Machine Learning (ML) algorithms automate or inform consequential decisions about people, individual decision subjects are often incentivized to strategically modify their observable attributes to receive more favorable predictions. As a result, the distribution the assessment rule is trained on may differ from the one it operates on in deployment. While such distribution shifts, in general, can hinder accurate predictions, our work identifies a unique opportunity associated with shifts due to strategic responses: We show that we can use strategic responses effectively to recover causal relationships between the observable features and outcomes we wish to predict, even under the presence of unobserved confounding variables. Specifically, our work establishes a novel connection between strategic responses to ML models and instrumental variable (IV) regression by observing that the sequence of deployed models can be viewed as an instrument that affects agents’ observable features but does not directly influence their outcomes. We show that our causal recovery method can be utilized to improve decision-making across several important criteria: individual fairness, agent outcomes, and predictive risk. In particular, we show that if decision subjects differ in their ability to modify non-causal attributes, any decision rule deviating from the causal coefficients can lead to (potentially unbounded) individual-level unfairness..


> Poster: A Single-Loop Gradient Descent and Perturbed Ascent Algorithm for Nonconvex Functional Constrained Optimization
> 
> Authors: Songtao Lu
> 
