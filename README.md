# ReachablePoint
a method to calculate reachablepoint based on deep learning

A reachable point means it has the largst viewshed in an area. The method proposed a network called GxyNet to generates a mapping between height to viewshed, the reuslt shows the height heatmap of every point and 
the viewshed heatmap together.

# Project Structure
***config***: use to manage the hypeparameters of the working process  
***data***: use to store raw datasets, geneate dataset and dataloader scripts  
***modeling***: define the model of the task, here is the GxyNet  
***runs***: contains train and test files to run project, also store the epoch model in logs  
***test_output***: store the running result  
***utils***: some self-defing functions may used in project



