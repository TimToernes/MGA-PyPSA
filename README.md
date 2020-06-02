## The near-optimal feasible space characteristics of energy-economic models

This repository contains the master theisis of Tim Pedersen 

Authour: Tim Pedersen

Superviser: Gorm B. Andresen 

Aarhus University, Denmark


## Abstract 
To limit the extent of irreversible climate change and accepting public opinion expressed in Greta Thunberg's speech at the UN Climate Action summit, drastic measures are needed to reduce the emission of greenhouse gases. The majority of CO2 emitted by humans are results of energy production to cover the ever-rising energy demand including transportation, heating, and electricity. To assist scientists and policymakers in their stride to reach ambitious goals in the reduction of CO2 emissions, analysis tools must be developed. An important tool, when it comes to planning of global and local energy networks are numeric energy-economic models. These models are capable of providing great insight into complex systems such as the European electricity grid and allows the user to make predictions about future needs and design strategies. 

Numeric energy-economic models do however suffer from great uncertainties related to uncertainties in input data called parametric uncertainty, and uncertainties arising from flaws in the mathematical formulation and construction of the energy-economic model referred to as structural uncertainty. 
If the uncertainties are not addressed, the results become untrustworthy and ends up providing little to no insight.
Several techniques exist to address parametric uncertainty, such as sensitivity analysis and Monte Carlo simulations. But until recently, no methods for addressing the structural uncertainty of the energy-economic models existed. This changed in 2010 when J. DeCarolis published a paper proposing a technique called "Modeling to generate alternatives (MGA)" doing just so. \hl{The root cause of structural uncertainty cannot be addressed as it can with parametric uncertainty, as the origin of structural uncertainty is hard to define. Instead one must investigate all solutions near the one found to be optimal, and estimate the likelihood of these near optimal solutions being the true optimal solution. In the technique proposed by J. DeCarolis, a finite set of maximally different near-optimal solutions are found. The difference in the found solutions can then be used as a measure of structural uncertainty and provides a variety of alternatives to the optimal configuration of the energy system. }

The proposed technique by J. DeCarolis does however, suffer from a range of flaws, arising from lacking structure in the manner near-optimal solutions are found. To obtain a complete picture of all near optimal solutions, a structured method of finding these is needed. 
The objective of this thesis is to explore the characteristics of all near optimal solutions contained withing the near-optimal feasible decision space, and to develop a new technique that in a structured manner can explore all solutions located within this space.  

Analysis of the common mathematical formulation of numeric energy-economic models that the model consists of linear constraints and therefore, the near-optimal feasible decision space, containing all near optimal solutions to the model, must be convex. 
Knowing these properties, a technique has been developed capable of searching the entire near optimal feasible decision space. The technique iteratively converges towards the full solution, and provides statistical information about the composition of the true optimal solution to the model. 
Furthermore, a method reducing the complexity of the mathematical problem, by grouping of variables is proposed. Grouping the variables in the model to form a new set of variables, does however reduce the amount of information obtained by solving this simplified problem. The effects of grouping the model variables is explored, and the effect is found to be significant, but predictable. 

The developed techniques is applied to a model of the European electricity grid. The usefulness of the techniques are proven as they provides information about the distribution of technology capacities in all near optimal solutions to the used model of the European electricity gird. 
