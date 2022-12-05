# TensorFlow Data Validation：检查和分析您的数据

一旦您的数据进入 TFX 管道，您就可以使用 TFX 组件对其进行分析和转换。您甚至可以在训练模型之前使用这些工具。

分析和转换数据的原因有很多：

- 查找数据中的问题。常见问题包括：
    - 缺少数据，例如具有空值的特征。
    - 标签被视为特征，以便您的模型在训练期间能够看到正确的答案。
    - 值超出预期范围的特征。
    - 数据异常。
    - 迁移学习模型的预处理与训练数据不匹配。
- 设计更有效的功能集。例如，您可以识别：
    - 特别是信息功能。
    - 冗余特征。
    - 规模差异如此之大的特征可能会减慢学习速度。
    - 具有很少或没有独特预测信息的特征。

TFX 工具既可以帮助发现数据错误，也可以帮助进行特征工程。

## TensorFlow 数据验证

- [概述](#overview)
- [基于模式的示例验证](#schema_based_example_validation)
- [训练-服务偏差检测](#skewdetect)
- [漂移检测](#drift_detection)

### 概述

TensorFlow Data Validation 识别训练和服务数据中的异常，并可以通过检查数据自动创建模式。该组件可以配置为检测数据中不同类别的异常。它可以

1. 通过将数据统计与编纂用户期望的模式进行比较来执行有效性检查。
2. 通过比较训练和服务数据中的示例来检测训练-服务偏差。
3. 通过查看一系列数据来检测数据漂移。

我们独立记录这些功能中的每一个：

- [基于模式的示例验证](#schema_based_example_validation)
- [训练-服务偏差检测](#skewdetect)
- [漂移检测](#drift_detection)

### 基于模式的示例验证

TensorFlow Data Validation 通过将数据统计与模式进行比较来识别输入数据中的任何异常。该模式编码了输入数据预期满足的属性，例如数据类型或分类值，并且可以由用户修改或替换。

Tensorflow 数据验证通常在 TFX 管道的上下文中被多次调用：(i) 对于从 ExampleGen 获得的每个拆分，(ii) 对于 Transform 使用的所有预转换数据，以及 (iii) 对于由 Transform 生成的所有转换后数据转换。在转换 (ii-iii) 的上下文中调用时，可以通过定义[`stats_options_updater_fn`](tft.md)来设置统计选项和基于模式的约束。这在验证非结构化数据（例如文本特征）时特别有用。有关示例，请参见[用户代码](https://github.com/tensorflow/tfx/blob/master/tfx/examples/bert/mrpc/bert_mrpc_utils.py)。

#### 高级架构功能

本节涵盖更高级的架构配置，可帮助进行特殊设置。

##### 稀疏特征

在示例中编码稀疏特征通常会引入多个预期对所有示例具有相同效价的特征。例如稀疏特征：

<pre><code>
WeightedCategories = [('CategoryA', 0.3), ('CategoryX', 0.7)]
</code></pre>

将使用单独的索引和值特征进行编码：

<pre><code>
WeightedCategoriesIndex = ['CategoryA', 'CategoryX']
WeightedCategoriesValue = [0.3, 0.7]
</code></pre>

限制条件是索引和值特征的效价应与所有示例匹配。可以通过定义 sparse_feature 在模式中明确表示此限制：

<pre><code class="lang-proto">
sparse_feature {
  name: 'WeightedCategories'
  index_feature { name: 'WeightedCategoriesIndex' }
  value_feature { name: 'WeightedCategoriesValue' }
}
</code></pre>

稀疏特征定义需要一个或多个索引和一个值特征，它们引用模式中存在的特征。明确定义稀疏特征使 TFDV 能够检查所有参考特征的效价是否匹配。

一些用例在特征之间引入了类似的效价限制，但不一定对稀疏特征进行编码。使用稀疏特性应该可以解除阻塞，但并不理想。

##### 架构环境

默认情况下，验证假定管道中的所有示例都遵循单个模式。在某些情况下，引入轻微的模式变化是必要的，例如在训练期间需要用作标签的特征（并且应该被验证），但在服务期间缺失。环境可用于表达此类要求，特别是`default_environment()` 、 `in_environment()` 、 `not_in_environment()` 。

例如，假设训练需要一个名为“LABEL”的特征，但预计在服务中会缺失。这可以表示为：

- 在模式中定义两个不同的环境：["SERVING", "TRAINING"] 并将“LABEL”仅与环境“TRAINING”相关联。
- 将训练数据与环境“TRAINING”相关联，将服务数据与环境“SERVING”相关联。

##### 架构生成

输入数据模式被指定为 TensorFlow [Schema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto)的一个实例。

开发人员可以依赖 TensorFlow Data Validation 的自动架构构建，而不是从头开始手动构建架构。具体来说，TensorFlow Data Validation 根据管道中可用训练数据计算的统计数据自动构建初始模式。用户可以简单地查看这个自动生成的模式，根据需要对其进行修改，将其签入版本控制系统，并将其显式推送到管道中以进行进一步验证。

TFDV 包括`infer_schema()`以自动生成模式。例如：

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```

这将触发基于以下规则的自动架构生成：

- 如果模式已经自动生成，则按原样使用。

- 否则，TensorFlow Data Validation 会检查可用的数据统计信息并为数据计算合适的模式。

*注意：自动生成的模式是尽力而为的，只会尝试推断数据的基本属性。希望用户根据需要对其进行审查和修改。*

### Training-Serving Skew Detection<a name="skewdetect"></a>

#### Overview

TensorFlow Data Validation can detect distribution skew between training and serving data. Distribution skew occurs when the distribution of feature values for training data is significantly different from serving data. One of the key causes for distribution skew is using either a completely different corpus for training data generation to overcome lack of initial data in the desired corpus. Another reason is a faulty sampling mechanism that only chooses a subsample of the serving data to train on.

##### 示例场景

注意：例如，为了补偿代表性不足的数据切片，如果在没有适当增加下采样示例的情况下使用有偏采样，则训练数据和服务数据之间的特征值分布会人为倾斜。

有关配置训练-服务偏差检测的信息，请参阅[TensorFlow Data Validation 入门指南](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)。

### 漂移检测

在连续的数据跨度之间（即跨度 N 和跨度 N+1 之间）支持漂移检测，例如不同天数的训练数据之间。我们根据分类特征的[L 无穷大距离](https://en.wikipedia.org/wiki/Chebyshev_distance)和数字特征的近似[Jensen-Shannon 散度](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)表示漂移。您可以设置阈值距离，以便在漂移高于可接受范围时收到警告。设置正确的距离通常是一个需要领域知识和实验的迭代过程。

有关配置漂移检测的信息，请参阅[TensorFlow Data Validation 入门指南](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)。

## 使用可视化来检查您的数据

TensorFlow Data Validation 提供了可视化特征值分布的工具。通过使用[Facets](https://pair-code.github.io/facets/)在 Jupyter notebook 中检查这些分布，您可以发现数据的常见问题。

![功能统计](images/feature_stats.png)

### 识别可疑分布

您可以通过使用 Facets Overview 显示来查找特征值的可疑分布来识别数据中的常见错误。

#### 不平衡数据

不平衡特征是一个值占主导地位的特征。不平衡的特征可能会自然发生，但如果一个特征总是具有相同的值，则可能存在数据错误。要检测 Facets Overview 中的不平衡特征，请从“排序依据”下拉列表中选择“非均匀性”。

最不平衡的特征将列在每个特征类型列表的顶部。例如，下面的屏幕截图显示了“数字特征”列表顶部的一个全为零的特征和一个高度不平衡的第二个特征：

![不平衡数据的可视化](images/unbalanced.png)

#### 均匀分布的数据

均匀分布的特征是所有可能值以接近相同的频率出现的特征。与不平衡数据一样，这种分布可以自然发生，但也可能由数据错误产生。

要在 Facets Overview 中检测均匀分布的特征，请从“排序依据”下拉列表中选择“非均匀性”并选中“反向顺序”复选框：

![统一数据的直方图](images/uniform.png)

如果唯一值不超过 20 个，则字符串数据使用条形图表示，如果唯一值超过 20 个，则使用累积分布图表示。因此，对于字符串数据，均匀分布可以显示为像上面那样的扁平条形图或像下面这样的直线：

![折线图：均匀数据的累积分布](images/uniform_cumulative.png)

##### 可以产生均匀分布数据的错误

以下是一些可以产生均匀分布数据的常见错误：

- 使用字符串表示非字符串数据类型，例如日期。例如，对于日期时间特征，您将有许多唯一值，其表示形式如“2017-03-01-11-45-03”。唯一值将均匀分布。

- 包括像“行号”这样的索引作为特征。在这里，您再次拥有许多独特的价值。

#### 缺失数据

要检查一个特征是否完全缺失值：

1. 从“排序依据”下拉列表中选择“金额缺失/零”。
2. 选中“反向顺序”复选框。
3. 查看“缺失”列以查看具有特征缺失值的实例的百分比。

数据错误也可能导致不完整的特征值。例如，您可能期望某个特征的值列表始终包含三个元素，但发现有时它只有一个。要检查不完整的值或特征值列表没有预期元素数量的其他情况：

1. 从右侧的“要显示的图表”下拉菜单中选择“值列表长度”。

2. 查看每个功能行右侧的图表。该图表显示了特征值列表长度的范围。例如，下面屏幕截图中突出显示的行显示了一个具有一些零长度值列表的特征：

![具有零长度特征值列表的特征的分面概览显示](images/zero_length.png)

#### 要素之间的规模差异很大

如果您的特征在规模上差异很大，那么模型可能难以学习。例如，如果某些特征从 0 到 1 变化，而其他特征从 0 到 1,000,000,000 变化，则规模差异很大。比较各个要素的“最大”和“最小”列，以找到差异很大的比例。

考虑规范化特征值以减少这些广泛的变化。

#### 标签无效的标签

TensorFlow 的 Estimator 对其接受作为标签的数据类型有限制。例如，二元分类器通常只适用于 {0, 1} 标签。

查看 Facets Overview 中的标签值，确保它们符合[Estimators 的要求](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/feature_columns.md)。
