

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm

# ==========================================================================
# Functions to set up the variables used implicitly in the functions
# ==========================================================================

def set_feature_names(x):
  global feature_names
  feature_names = x

def set_dictionary(x):
  global dictionary
  dictionary = x

def set_texts(x):
  global texts
  texts = x

def set_top_words(x):
  global top_n
  top_n = x

def set_vectorizer(x):
  global vectorizer
  vectorizer = x



# ==========================================================================
# Functions to compute the metrics
# ==========================================================================

def get_topics(model, bert=False):
  """
  Extracts topics from a topic model.
  Args:
      model (any): The model.
      bert (bool, optional): Whether the model is from sklearn or a BERTopic model. Defaults to False.
  Returns:
      list: A list with the topics.
  Notes:
      - This function uses the global variables `feature_names` and `top_n`,
        make sure they are set in the notebook before calling this function.
  """
  topics = []
  if not bert:
    for topic_weights in model.components_:
        top_features = [feature_names[i] for i in topic_weights.argsort()[-top_n:]]
        topics.append(top_features)
  else:
    for topic_id, topic in model.get_topics().items():
        if topic_id != -1:
          topics.append([word for word, _ in topic[:top_n]])

  return topics



def get_diversity(model, bert=False):
  """
  Computes the diversity score for a topic model.
  Args:
      model (any): The model
      bert (bool, optional): Whether the model is from sklearn or a BERTopic model. Defaults to False.
  Returns:
      float: The diversity score.
  Notes:
    - This function uses the global variable `vectorizer`,
      make sure it is set in the notebook before calling this function.
  """
  topics = get_topics(model, bert=bert)
  topics = [vectorizer.transform([' '.join(word for word in topic)]).toarray() for topic in topics]
  cos_sim = cosine_similarity(np.concatenate(topics, axis=0))[~np.eye(len(topics), dtype=bool)]

  return 1 - np.mean(cos_sim)



def get_coherence(model, bert=False):
  """
  Computes the coherence score for a topic model.
  Args:
      model (any): The model.
      bert (bool, optional): Whether the model is from sklearn or a BERTopic model. Defaults to False.
  Returns:
      float: The coherence score.
  Notes:
    - This function uses the global variables`texts` and `dictionary`,
      make sure they are set in the notebook before calling this function.
  """
  from gensim.models.coherencemodel import CoherenceModel
  topics = get_topics(model) if not bert else get_topics(model, bert=True)
  coherence_model = CoherenceModel(topics = topics,
                                   texts = texts,
                                   dictionary = dictionary,
                                   coherence = 'c_v'
                                   )

  return coherence_model.get_coherence()



def get_rouge(y_test, predicted):
  """
  Computes F1 scores for ROUGE-1, ROUGE-2, ROUGE-L for a summarized text.
  Args:
      y_test (string): The real summary.
      predicted (string): The summarized doc.
  Returns:
      tuple: A tuple of three floats (rouge1_f1, rouge2_f1, rougeL_f1).
  """
  from rouge_score import rouge_scorer
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer = True)
  scores = scorer.score(y_test, predicted)

  return scores['rouge1'][2], scores['rouge2'][2], scores['rougeL'][2]



# ==========================================================================
# Functions for the tuning process
# ==========================================================================

def create_space(param_grid):
  """
  Creates a parameter space from a parameter grid.
  Args:
      param_grid (dict): Dictionary mapping parameter names (str) to lists of possible values.
  Yields:
      dict: A dictionary representing one combination of parameters from the grid.
  """
  keys = param_grid.keys()
  for values in itertools.product(*param_grid.values()):
    yield dict(zip(keys, values))



def tuning_tester(model, param_grid, data):
  """
  Evaluates a sklearn model on all hyperparameter combinations and computes metrics over multiple seeds.
  Args:
      model (any): The model.
      param_grid (dict): Dictionary mapping parameter names (str) to lists of possible values.
      data (any): Dataset to fit the model on (format depends on the model).
  Returns:
      dict: A dictionary containing lists of parameter values and corresponding evaluation metrics:
                  - Keys from `param_grid` contain the specific hyperparameter values for each combination
                  - "diversity": Mean diversity score over seeds
                  - "diversity_std": Standard deviation of diversity over seeds
                  - "coherence": Mean coherence score over seeds
                  - "coherence_std": Standard deviation of coherence over seeds
  """
  results = {x:[] for x in param_grid.keys()}
  results.update({x:[] for x in ["diversity", "diversity_std", "coherence", "coherence_std"]})

  for params in tqdm(create_space(param_grid), total=math.prod(map(len, param_grid.values())), desc='Evaluating model'):
    diversity, coherence = [], []

    for seed in tqdm(range(3), leave=False, desc='Evaluating seed'):
      model.set_params(**params, random_state=seed).fit(data)
      diversity.append(get_diversity(model))
      coherence.append(get_coherence(model))

    for key, value in params.items(): results[key].append(value)
    results["diversity"].append(np.mean(diversity))
    results["diversity_std"].append(np.std(diversity))
    results["coherence"].append(np.mean(coherence))
    results["coherence_std"].append(np.std(coherence))

  return results



def set_params(model, param_grid, best_iter):
  """
  Sets the hyperparameters of a sklearn model to the best combination based on a given index.
  Args:
      model (any): The model.
      param_grid (dict): Dictionary mapping parameter names (str) to lists of possible values.
      best_iter (int): The index of the best combination.
  Returns:
      any: The same model instance with updated hyperparameters and random_state fixed to 15.
  """
  best_params = {}
  for key, value in param_grid.items():
    best_params[key] = value[best_iter]
  model.set_params(**best_params, random_state=15)
  return model



# ==========================================================================
# Functions to plot the results
# ==========================================================================

def plot_top_words(model, title):
  """
  Plots the top words for 10 topics from a fitted sklearn topic model.
  Args:
      model (any): The fitted sklearn topic model.
      title (str): Title for the entire figure.
  Returns:
      None: Displays a matplotlib figure.
  Notes:
      - This function uses the global variables `feature_names` and `top_n`,
        make sure they are set in the notebook before calling this function.
  """
  fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
  axes = axes.flatten()

  for topic_idx, topic in enumerate(model.components_[:10]):
      top_features_ind = topic.argsort()[-top_n:]
      top_features = feature_names[top_features_ind]
      weights = topic[top_features_ind]

      ax = axes[topic_idx]
      ax.barh(top_features, weights, height=0.7)
      ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
      ax.tick_params(axis="both", which="major", labelsize=20)
      for i in "top right left".split():
          ax.spines[i].set_visible(False)
      fig.suptitle(title, fontsize=40)

  plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
  plt.show()



def tuning_plotter(results):
  """
  Plots coherence and diversity metrics with error bands from hyperparameter tuning results.
  Args:
      results (dict): Dictionary containing evaluation metrics for hyperparameter combinations.
  Returns:
      None: Displays a matplotlib figure.
  """
  fig, axes = plt.subplots(2, 1, sharex=True)
  metric, color = ['coherence', 'diversity'], ['b', 'g']

  for x in range(2):
    ax = axes[x]
    std_up = [results[metric[x]][j] + results[metric[x]+'_std'][j] for j in range(len(results[metric[x]]))]
    std_down = [results[metric[x]][j] - results[metric[x]+'_std'][j] for j in range(len(results[metric[x]]))]
    ax.plot(results[metric[x]], color=color[x], linewidth=1)
    ax.fill_between(range(len(results[metric[x]])), std_down, std_up, alpha=0.15, color = color[x])
    ax.set_xticks(range(len(results[metric[x]])))
    axes[1].set_xlabel("Hyperparameter combination")
    ax.set_title(metric[x].title())
    ax.grid()

  plt.show()



def plot_metrics(models, labels):
  """
  Plots coherence and diversity metrics for multiple models as a grouped bar chart.
  Args:
      models (list of tuples): Each tuple contains:
          - model: a fitted topic model
          - bert (bool, optional): Whether the model is from sklearn or a BERTopic model.
      labels (list of strings): Names of the models, used for the legend.
  Returns:
      None: Displays a matplotlib figure.
  """
  metrics = ['Coherence', 'Diversity']
  coherences = [get_coherence(model, bert=idx) for model,idx in models]
  diversities = [get_diversity(model, bert=idx) for model,idx in models]

  w = 0.1
  x= np.arange(len(metrics))*0.75
  fig, ax = plt.subplots()
  color = ['C0', 'C1', 'C2', 'C3']
  for i in range(len(labels)):
    ax.bar(x+i*w, [coherences[i], diversities[i]],
           width=w, label=labels[i], color=color[i])

  ax.set_xticks(x + w * (4 - 1) / 2)
  ax.set_xticklabels(metrics)
  ax.set_ylabel('Score')
  ax.legend()

  plt.show()



def plot_rouge(summaries, labels, summary_labels):
  """
  Plots F1 scores for ROUGE-1, ROUGE-2 and ROUGE-L with error bars for multiple summarizd docs.
  Args:
      summaries (list of list of str): a list of all the summaries for each method.
      labels (list of str): Names of the summarization methods.
      summary_labels (list of str): Real summaries.
  Returns:
      None: Displays a matplotlib figure.
  """
  metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
  results = {}

  for i in range(len(labels)):
    results[labels[i]] = dict(zip(metrics, [[],[],[]]))
    for j in range(len(summary_labels)):
      score = get_rouge(summary_labels[j], summaries[i][j])
      for k in range(3): results[labels[i]][metrics[k]].append(score[k])
    for m in metrics: results[labels[i]][m] = np.mean(results[labels[i]][m])

  w, x = 0.2, np.arange(len(metrics))
  fig, ax = plt.subplots()
  color = ['C9', 'C4', 'C8', 'C6']
  for i in range(4):
    ax.bar(x+i*w, [results[labels[i]][metric] for metric in metrics],
            width=w, label=labels[i], color=color[i])

  color = ['red', 'green']
  for i in range(4,6):
    pos = 0
    for metric in metrics:
      ax.axhline(y=results[labels[i]][metric], xmin=pos, xmax=pos+1/3,
                color=color[i-4], linestyle="--", linewidth=1)
      pos += 1/3

  ax.set_xticks(x + w * (4 - 1) / 2)
  ax.set_xticklabels(metrics)
  ax.set_ylabel('Score')
  leg1 = ax.legend(bbox_to_anchor=(3/4, 1.0))
  ax.add_artist(leg1)
  line_handles = [
      Line2D([0], [0], color='green', linestyle='--', linewidth=1),
      Line2D([0], [0], color='red', linestyle='--', linewidth=1)
  ]
  ax.legend(line_handles, ['Best', 'Random'], loc='upper right')

  plt.show()



