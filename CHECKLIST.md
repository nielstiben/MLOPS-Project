### Exercises

1. (Optional) Familiar yourself with each of the libraries. One way to do this is to find relevant tutorials on each project
   and try to figure out the code. Such tutorials will give you a rough idea how the API for each library looks like.

2. Form groups! The recommended group size is 4 persons, but we also accept 3 or 5 man groups. Try to find other people based
   on what framework that you would like to work with.

3. Brainstorm projects! Try to figure out exactly what you want to work with and especially how you are going to incorporate
   the frameworks (we are aware that you are not familiar with every framework yet) that you have chosen to work with into
   your project. The **Final exercise** for today is to formulate a project description (see bottom of this page).

4. When you formed groups and formulated a project you are allowed to start working on the actual code. I have included a
   to-do list at the bottom that somewhat summaries what we have done in the course until know. You are **NOT** expected
   to fulfill all bullet points on the to-do list today, as you will continues to work on the project in the following two weeks.

### Final objective

Final exercise for today is making a project description. Write around half to one page about:

* Overall goal of the project
* What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
* How to you intend to include the framework into your project
* What data are you going to run on (initially, may change)
* What deep learning models do you expect to use

The project description will serve as an guideline for us at the exam that you have somewhat reached the goals that you set out to do.

By the end of the day (17:00) you should upload your project description (in the `README.md` file) + whatever you have done on the project
until now to a github repository. When this you have done this, on DTU Learn go to assignments and hand in (as a group) the project description.

We will briefly (before next Monday) look over your github repository and project description to check that everything is fine. If we have
any questions/concerns we will contact you.

## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very
point on the checklist for the exam.

### Week 1

- [X] Create a git repository
- [X] Make sure that all team members have write access to the github repository
- [X] Create a dedicated environment for you project to keep track of your packages (using conda)
- [X] Create the initial file structure using cookiecutter
- [X] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
- [X] Add a model file and a training script and get that running
- [X] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [X] Remember to comply with good coding practices (`pep8`) while doing the project
- [X] Do a bit of code typing and remember to document essential parts of your code
- [X] Setup version control for your data or part of your data
- [X] Construct one or multiple docker files for your code
- [X] Build the docker files locally and make sure they work as intended -> (training works assuming data has been downloaded and processed)
- [X] Write one or multiple configurations files for your experiments
- [X] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [X] Use wandb to log training progress and other important metrics/artifacts in your code -> (should be checked)
- [X] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

- [X] Write unit tests related to the data part of your code
- [X] Write unit tests related to model construction
- [X] Calculate the coverage.
- [X] Get some continues integration running on the github repository
- [X] (optional) Create a new project on `gcp` and invite all group members to it
- [X] Create a data storage on `gcp` for you data
- [X] Create a trigger workflow for automatically building your docker images
- [X] Get your model training on `gcp` -> (training without GPU currently)
- [ ] Play around with distributed data loading
- [ ] (optional) Play around with distributed model training
- [X] Play around with quantization and compilation for you trained models

### Week 3

- [ ] Deployed your model locally using TorchServe
- [ ] Checked how robust your model is towards data drifting
- [X] Deployed your model using `gcp`
- [X] Monitored the system of your deployed model
- [X] Monitored the performance of your deployed model

### Additional

- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Make sure all group members have a understanding about all parts of the project
- [ ] Create a presentation explaining your project
- [X] Uploaded all your code to github
- [X] (extra) Implemented pre-commit hooks for your project repository
- [ ] (extra) Used Optuna to run hyperparameter optimization on your model

## Exam and Presentation

The exam includes:
*	6 min presentation
*	10 min discussion

For the presentation we are going to keep a fairly strict format to get the exam rolling.
Therefore, please create a presentation with the following 5 slides (+- 1 slide if you need it):

1.	Problem description: What problem is your model trying to solve?
2.	Model description: What kind of model are you using?
3.	Data description: What does your data look like (where did you get it from, size)?
4.	Framework: How did you include the framework that you choose to work with?
5.	Use case: Show something from the course that you think you did very well!
      As this is a practical course you are also free to give a live demo. Examples:

    * Did you really use the cookiecutter structure?
    * Did you make good use of `gcp` for your project?
    * Show (live) that you have deployed your model

The last slide will be used as springboard to talk about how you have used all the other
tools taught in the course. Please have both your presentation and webpage with your project
github repository and the main `gcp` account used ready before the exam so we can keep the
time plan.
