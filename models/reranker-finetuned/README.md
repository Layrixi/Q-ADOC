---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:5452
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-classification
      name: Cross Encoder Classification
    dataset:
      name: baseline
      type: baseline
    metrics:
    - type: accuracy
      value: 0.8221757322175732
      name: Accuracy
    - type: accuracy_threshold
      value: 2.870133876800537
      name: Accuracy Threshold
    - type: f1
      value: 0.6604477611940298
      name: F1
    - type: f1_threshold
      value: 0.4924195110797882
      name: F1 Threshold
    - type: precision
      value: 0.5959595959595959
      name: Precision
    - type: recall
      value: 0.7405857740585774
      name: Recall
    - type: average_precision
      value: 0.6604808520261856
      name: Average Precision
  - task:
      type: cross-encoder-classification
      name: Cross Encoder Classification
    dataset:
      name: finetuned
      type: finetuned
    metrics:
    - type: accuracy
      value: 0.893305439330544
      name: Accuracy
    - type: accuracy_threshold
      value: 1.7007057666778564
      name: Accuracy Threshold
    - type: f1
      value: 0.7940074906367041
      name: F1
    - type: f1_threshold
      value: -2.8201608657836914
      name: F1 Threshold
    - type: precision
      value: 0.7186440677966102
      name: Precision
    - type: recall
      value: 0.8870292887029289
      name: Recall
    - type: average_precision
      value: 0.8090976707955442
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['What magical spells do spellcasters gain access to at the start of the second tier in 11D&D Player’s Basic Rules v0.2?', 'be cast at a higher level. Magic permeates the worlds of D&D and most often appears in the form of a spell. This chapter provides the rules for casting spells. Different character classes have distinctive ways of learning and preparing their spells, and monsters use spells in unique ways. Regardless of its source, a spell follows the rules here. What Is a Spell? A spell is a discrete magical effect, a single shaping of the magical energies that suffuse the multiverse into a specific, limited expression. In casting a spell, a character carefully plucks at the invisible strands of raw magic suffusing the world, pins them in place in a particular pattern, sets them vibrating in a specific way, and then releases them to unleash the desired effect—in most cases, all in the span of seconds. Spells can be versatile tools, weapons, or protective wards. They can deal damage or undo it, impose or remove conditions (see appendix A), drain life energy away, and restore life to the dead. Uncounted thousands of spells have been created over the course of the multiverse’s history, and many of them are long forgotten. Some might yet lie recorded in crumbling spellbooks hidden in ancient ruins or trapped in the minds of dead gods. Or they might someday be reinvented by a character who has amassed enough power and wisdom to do so. Spell Level Every spell has a level from 0 to 9. A spell’s level is a general indicator of how powerful it is, with the lowly (but still impressive) magic missile at 1st level and the incredible time stop at 9th. Cantrips—simple but powerful spells that characters can cast almost by rote— are level 0. The higher a spell’s level, the higher level a spellcaster must be to use that spell.'],
    ['What damage resistances does the Adaptive Resistance coat grant, and for how long?', "one hand and no other weapons, you gain a +2 bonus to damage rolls with that weapon. TWO-WEAPON FIGHTING When you engage in two-weapon fighting, you can add your ability modifier to the damage of the second attack. SPELLCASTING By the time you reach 2nd levei, you have learned to use the magical essence of nature to cast spells, much as a druid does. See chapter 10 for the general rules of spellcasting and chapter 11 for the ranger spell Iist. SPELL SLOTS The Ranger table shows how many spell slots you have to cast your spells of 1st leveI and higher. To cast one of these spells, you must expend a slot of the spell's leveI or higher. Vou regain ali expended spell slots when you finish a long rest. For example, ifyou know the 1st-levei spell animal friendship and have a 1st-levei and a 2nd-leveI spell slot available, you can cast animal friendship using either slot. SPELLS KNOWN OF 1ST LEVE L AND HIGHER Vou know two 1st-levei spells ofyour choice from the ranger spelllist. The Spells Known column of the Ranger table shows when you learn more ranger spells of your choice. Each of these spells must be of a levei for which you have 9' spell slots. For instance, when you reach 5th levei in this c1ass, you can learn one new spell of 1st or 2nd leveI. Additionally, when you gain a levei in this c1ass, you can choose one of the ranger spells you know and replace it with another spell from the ranger spelllist, which also must be of a leveI for which you have spell slots. SPELLCASTING ABILITY Wisdom is your spellcasting ability for your ranger spells, since your magic draws on your attunement to nature. Vou use your Wisdom whenever a"],
    ['At what level do I gain additional warlock cantrips and spells?', "pouch or (b) an arcane focus (a)a scholar's pack or (b) a dungeoneer's pack Lealher armor, any simple weapon, and lwo daggers OTHERWORLDLY PATRON AI Isllevel, you have slruck a bargain wilh an olherworldly being ofyour choice: lhe Archfey, lhe Fiend, or lhe Greal Old One, each ofwhich is delailed aI lhe end of lhe class descriplion, Your choice granls you fealures aI Isllevel and again aI 61h, 101h, and 141hleveI. PACT MAGIC Your arcane research and lhe magic beslowed on you byyour palron have given you facilily wilh spells. See chapter 10for lhe general rules of spellcasling and chapter 11for lhe warlock spelllisl. CANTRIPS Vouknow lwo canlrips ofyour choice from lhe warlock spelllisl. Voulearn addilional warlock canlrips ofyour choice aI higher leveis, as shown in lhe Canlrips Known column of lhe Warlock lable. SPELL SLOTS The Warlock lable shows how many spell slols you have. The lable also shows whallhe leveIoflhose slols is;ali ofyour spell slols are lhe same leveI.Tocasl one ofyour warlock spells of Isllevel or higher, you musl expend a spell slol. Vouregain ali expended spell slols when you finish a shorl or long resl. For example, when you are 51hleveI,you have lwo 3rd-level spell sioIs. To casllhe Isl-level spell thunderwave, you musl spend one of lhose sioIs, and you caSl il as a 3rd-level spell. SPELLS KNOWN OF 1ST LEVEL AND HIGHER AI Isllevel, you know lwo Isl-level spells ofyour choice from lhe warlock spelllisl. The Spells Known column of lhe Warlock lable shows when you learn more warlock spells ofyour choice of ls1 levei and higher. Aspell you choose musl be of a levei no higher lhan whal's shown in lhe lable's Slol LeveI column for your leveI.When you reach 61h levei, for example, you learn a new warlock spell, which can be ISI, 2nd,"],
    ['Does using Dominate Monster allow me to make the target use its reaction, and if so, what do I need to do?', "such as a fire or pit, but itwill provoke opportunity attacks to move in the designated direction. CONE OF COLD 5th-levei evocation Casting Time: I action Range: 5elf (60-foot cone) Components: V,5, M(asmall crystal or glass cone) Duration: Instantaneous Ablast ofcold air erupts from your hands. Each creature in a 60-foot cone must make a Constitution saving throw. Acreature takes 8d8 cold damage on a failed save, or half as much damage on a successful one. Acreature killed bythis spell becomes a frozen statue until itthaws. At Higher LeveIs. When you cast this speIl using a spell slot of6th level or higher, the damage increases by ld8 foreach slot leveIabove 5th. CONFUSION 4th-levei enchantment Casting Time: 1action Range: 90 feet Components: V,5, M(three nut shells) Duration: Concentration, up to I minute This spell assaults and twists creatures' minds, spawning delusions and provoking uncontrolled action. Each creature in a lO-foot-radius sphere centered on a point you choose within range must succeed on a Wisdom saving throw when you cast this spell or be affected byit. An affected target can't take reactions and must roIl adIO at the start ofeach ofits turns to determine its behavior forthat turno dl0 Behavior 1 The creature uses ali its movement to move in a random direction. To determine the direction, reli a d8 and assign a direction to each die face. The creature doesn't take an action this turno 2-6 The creature doesn't move or take actions this turno 7-8 The creature uses its action to make a melee attack against a randomly determined creature within its reach. If there is no creature within its reach, the creature does nothing this turno 9-10 The creature can act and move normally. At the end ofeach of its turns, an affected target can make a Wisdom saving"],
    ['What proficiencies does a tamer gain when taking their first level as a tamer?', 'with that weapon type . Proficiencies Gained. If tamer isn’t your initial class, you gain the following proficiencies when you take your first level as a tamer: light armour, shields, simple weapons, and nets . Spell Slots. Add half your levels (rounded down) in the tamer class to the appropriate levels from oth- er classes to determine your available spell slots . Pocket Familiar 1st-level Tamer feature Y ou become bonded to a companion that accompa- nies you on your adventures and is trained to fight alongside you . Choose a Small or smaller creature with a challenge rating of 1/2 or lower that isn’t a humanoid, giant, or swarm to become your com- panion . When a creature becomes your companion, it has a maximum number of hit points equal to the average of its Hit Dice, as indicated in its statistics, and it can’t cast spells. Work with your GM to find a companion that suits your campaign world . This companion obeys your commands and is friendly to you and your allies . V essel. When not summoned, your companion exists inside a magical vessel of your own design, such as a painted animal skull, bejewelled egg, or crystal sphere . While in this vessel, the companion has full cover from all attacks and other effects, is unaffected by area of effects that originate from outside the ves- sel, and exists in stasis; it doesn’t need to eat, drink, sleep, or breathe, and it is immune to poison and dis- ease, although a poison or disease already in its system is suspended, not neutralised . A companion at 0 hit points is instantly stabilised when it enters its vessel . If a vessel is broken, or a companion is released from its vessel for any other reason,'],
]
scores = model.predict(pairs)
print(scores)
# [ -8.9652 -10.9546   2.9938  -8.595    7.9582]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What magical spells do spellcasters gain access to at the start of the second tier in 11D&D Player’s Basic Rules v0.2?',
    [
        'be cast at a higher level. Magic permeates the worlds of D&D and most often appears in the form of a spell. This chapter provides the rules for casting spells. Different character classes have distinctive ways of learning and preparing their spells, and monsters use spells in unique ways. Regardless of its source, a spell follows the rules here. What Is a Spell? A spell is a discrete magical effect, a single shaping of the magical energies that suffuse the multiverse into a specific, limited expression. In casting a spell, a character carefully plucks at the invisible strands of raw magic suffusing the world, pins them in place in a particular pattern, sets them vibrating in a specific way, and then releases them to unleash the desired effect—in most cases, all in the span of seconds. Spells can be versatile tools, weapons, or protective wards. They can deal damage or undo it, impose or remove conditions (see appendix A), drain life energy away, and restore life to the dead. Uncounted thousands of spells have been created over the course of the multiverse’s history, and many of them are long forgotten. Some might yet lie recorded in crumbling spellbooks hidden in ancient ruins or trapped in the minds of dead gods. Or they might someday be reinvented by a character who has amassed enough power and wisdom to do so. Spell Level Every spell has a level from 0 to 9. A spell’s level is a general indicator of how powerful it is, with the lowly (but still impressive) magic missile at 1st level and the incredible time stop at 9th. Cantrips—simple but powerful spells that characters can cast almost by rote— are level 0. The higher a spell’s level, the higher level a spellcaster must be to use that spell.',
        "one hand and no other weapons, you gain a +2 bonus to damage rolls with that weapon. TWO-WEAPON FIGHTING When you engage in two-weapon fighting, you can add your ability modifier to the damage of the second attack. SPELLCASTING By the time you reach 2nd levei, you have learned to use the magical essence of nature to cast spells, much as a druid does. See chapter 10 for the general rules of spellcasting and chapter 11 for the ranger spell Iist. SPELL SLOTS The Ranger table shows how many spell slots you have to cast your spells of 1st leveI and higher. To cast one of these spells, you must expend a slot of the spell's leveI or higher. Vou regain ali expended spell slots when you finish a long rest. For example, ifyou know the 1st-levei spell animal friendship and have a 1st-levei and a 2nd-leveI spell slot available, you can cast animal friendship using either slot. SPELLS KNOWN OF 1ST LEVE L AND HIGHER Vou know two 1st-levei spells ofyour choice from the ranger spelllist. The Spells Known column of the Ranger table shows when you learn more ranger spells of your choice. Each of these spells must be of a levei for which you have 9' spell slots. For instance, when you reach 5th levei in this c1ass, you can learn one new spell of 1st or 2nd leveI. Additionally, when you gain a levei in this c1ass, you can choose one of the ranger spells you know and replace it with another spell from the ranger spelllist, which also must be of a leveI for which you have spell slots. SPELLCASTING ABILITY Wisdom is your spellcasting ability for your ranger spells, since your magic draws on your attunement to nature. Vou use your Wisdom whenever a",
        "pouch or (b) an arcane focus (a)a scholar's pack or (b) a dungeoneer's pack Lealher armor, any simple weapon, and lwo daggers OTHERWORLDLY PATRON AI Isllevel, you have slruck a bargain wilh an olherworldly being ofyour choice: lhe Archfey, lhe Fiend, or lhe Greal Old One, each ofwhich is delailed aI lhe end of lhe class descriplion, Your choice granls you fealures aI Isllevel and again aI 61h, 101h, and 141hleveI. PACT MAGIC Your arcane research and lhe magic beslowed on you byyour palron have given you facilily wilh spells. See chapter 10for lhe general rules of spellcasling and chapter 11for lhe warlock spelllisl. CANTRIPS Vouknow lwo canlrips ofyour choice from lhe warlock spelllisl. Voulearn addilional warlock canlrips ofyour choice aI higher leveis, as shown in lhe Canlrips Known column of lhe Warlock lable. SPELL SLOTS The Warlock lable shows how many spell slols you have. The lable also shows whallhe leveIoflhose slols is;ali ofyour spell slols are lhe same leveI.Tocasl one ofyour warlock spells of Isllevel or higher, you musl expend a spell slol. Vouregain ali expended spell slols when you finish a shorl or long resl. For example, when you are 51hleveI,you have lwo 3rd-level spell sioIs. To casllhe Isl-level spell thunderwave, you musl spend one of lhose sioIs, and you caSl il as a 3rd-level spell. SPELLS KNOWN OF 1ST LEVEL AND HIGHER AI Isllevel, you know lwo Isl-level spells ofyour choice from lhe warlock spelllisl. The Spells Known column of lhe Warlock lable shows when you learn more warlock spells ofyour choice of ls1 levei and higher. Aspell you choose musl be of a levei no higher lhan whal's shown in lhe lable's Slol LeveI column for your leveI.When you reach 61h levei, for example, you learn a new warlock spell, which can be ISI, 2nd,",
        "such as a fire or pit, but itwill provoke opportunity attacks to move in the designated direction. CONE OF COLD 5th-levei evocation Casting Time: I action Range: 5elf (60-foot cone) Components: V,5, M(asmall crystal or glass cone) Duration: Instantaneous Ablast ofcold air erupts from your hands. Each creature in a 60-foot cone must make a Constitution saving throw. Acreature takes 8d8 cold damage on a failed save, or half as much damage on a successful one. Acreature killed bythis spell becomes a frozen statue until itthaws. At Higher LeveIs. When you cast this speIl using a spell slot of6th level or higher, the damage increases by ld8 foreach slot leveIabove 5th. CONFUSION 4th-levei enchantment Casting Time: 1action Range: 90 feet Components: V,5, M(three nut shells) Duration: Concentration, up to I minute This spell assaults and twists creatures' minds, spawning delusions and provoking uncontrolled action. Each creature in a lO-foot-radius sphere centered on a point you choose within range must succeed on a Wisdom saving throw when you cast this spell or be affected byit. An affected target can't take reactions and must roIl adIO at the start ofeach ofits turns to determine its behavior forthat turno dl0 Behavior 1 The creature uses ali its movement to move in a random direction. To determine the direction, reli a d8 and assign a direction to each die face. The creature doesn't take an action this turno 2-6 The creature doesn't move or take actions this turno 7-8 The creature uses its action to make a melee attack against a randomly determined creature within its reach. If there is no creature within its reach, the creature does nothing this turno 9-10 The creature can act and move normally. At the end ofeach of its turns, an affected target can make a Wisdom saving",
        'with that weapon type . Proficiencies Gained. If tamer isn’t your initial class, you gain the following proficiencies when you take your first level as a tamer: light armour, shields, simple weapons, and nets . Spell Slots. Add half your levels (rounded down) in the tamer class to the appropriate levels from oth- er classes to determine your available spell slots . Pocket Familiar 1st-level Tamer feature Y ou become bonded to a companion that accompa- nies you on your adventures and is trained to fight alongside you . Choose a Small or smaller creature with a challenge rating of 1/2 or lower that isn’t a humanoid, giant, or swarm to become your com- panion . When a creature becomes your companion, it has a maximum number of hit points equal to the average of its Hit Dice, as indicated in its statistics, and it can’t cast spells. Work with your GM to find a companion that suits your campaign world . This companion obeys your commands and is friendly to you and your allies . V essel. When not summoned, your companion exists inside a magical vessel of your own design, such as a painted animal skull, bejewelled egg, or crystal sphere . While in this vessel, the companion has full cover from all attacks and other effects, is unaffected by area of effects that originate from outside the ves- sel, and exists in stasis; it doesn’t need to eat, drink, sleep, or breathe, and it is immune to poison and dis- ease, although a poison or disease already in its system is suspended, not neutralised . A companion at 0 hit points is instantly stabilised when it enters its vessel . If a vessel is broken, or a companion is released from its vessel for any other reason,',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Classification

* Datasets: `baseline` and `finetuned`
* Evaluated with [<code>CrossEncoderClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator)

| Metric                | baseline   | finetuned  |
|:----------------------|:-----------|:-----------|
| accuracy              | 0.8222     | 0.8933     |
| accuracy_threshold    | 2.8701     | 1.7007     |
| f1                    | 0.6604     | 0.794      |
| f1_threshold          | 0.4924     | -2.8202    |
| precision             | 0.596      | 0.7186     |
| recall                | 0.7406     | 0.887      |
| **average_precision** | **0.6605** | **0.8091** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,452 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                            | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                                | float                                                          |
  | modality | text                                                                               | text                                                                                  |                                                                |
  | details  | <ul><li>min: 12 tokens</li><li>mean: 19.79 tokens</li><li>max: 40 tokens</li></ul> | <ul><li>min: 225 tokens</li><li>mean: 424.64 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.25</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What magical spells do spellcasters gain access to at the start of the second tier in 11D&D Player’s Basic Rules v0.2?</code> | <code>be cast at a higher level. Magic permeates the worlds of D&D and most often appears in the form of a spell. This chapter provides the rules for casting spells. Different character classes have distinctive ways of learning and preparing their spells, and monsters use spells in unique ways. Regardless of its source, a spell follows the rules here. What Is a Spell? A spell is a discrete magical effect, a single shaping of the magical energies that suffuse the multiverse into a specific, limited expression. In casting a spell, a character carefully plucks at the invisible strands of raw magic suffusing the world, pins them in place in a particular pattern, sets them vibrating in a specific way, and then releases them to unleash the desired effect—in most cases, all in the span of seconds. Spells can be versatile tools, weapons, or protective wards. They can deal damage or undo it, impose or remove conditions (see appendix A), drain life energy away, and restore life to the dead. Uncounted ...</code> | <code>0.0</code> |
  | <code>What damage resistances does the Adaptive Resistance coat grant, and for how long?</code>                                     | <code>one hand and no other weapons, you gain a +2 bonus to damage rolls with that weapon. TWO-WEAPON FIGHTING When you engage in two-weapon fighting, you can add your ability modifier to the damage of the second attack. SPELLCASTING By the time you reach 2nd levei, you have learned to use the magical essence of nature to cast spells, much as a druid does. See chapter 10 for the general rules of spellcasting and chapter 11 for the ranger spell Iist. SPELL SLOTS The Ranger table shows how many spell slots you have to cast your spells of 1st leveI and higher. To cast one of these spells, you must expend a slot of the spell's leveI or higher. Vou regain ali expended spell slots when you finish a long rest. For example, ifyou know the 1st-levei spell animal friendship and have a 1st-levei and a 2nd-leveI spell slot available, you can cast animal friendship using either slot. SPELLS KNOWN OF 1ST LEVE L AND HIGHER Vou know two 1st-levei spells ofyour choice from the ranger spelllist. The Spells Kn...</code> | <code>0.0</code> |
  | <code>At what level do I gain additional warlock cantrips and spells?</code>                                                        | <code>pouch or (b) an arcane focus (a)a scholar's pack or (b) a dungeoneer's pack Lealher armor, any simple weapon, and lwo daggers OTHERWORLDLY PATRON AI Isllevel, you have slruck a bargain wilh an olherworldly being ofyour choice: lhe Archfey, lhe Fiend, or lhe Greal Old One, each ofwhich is delailed aI lhe end of lhe class descriplion, Your choice granls you fealures aI Isllevel and again aI 61h, 101h, and 141hleveI. PACT MAGIC Your arcane research and lhe magic beslowed on you byyour palron have given you facilily wilh spells. See chapter 10for lhe general rules of spellcasling and chapter 11for lhe warlock spelllisl. CANTRIPS Vouknow lwo canlrips ofyour choice from lhe warlock spelllisl. Voulearn addilional warlock canlrips ofyour choice aI higher leveis, as shown in lhe Canlrips Known column of lhe Warlock lable. SPELL SLOTS The Warlock lable shows how many spell slols you have. The lable also shows whallhe leveIoflhose slols is;ali ofyour spell slols are lhe same leveI.Tocasl one ofyo...</code> | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss | baseline_average_precision | finetuned_average_precision |
|:------:|:----:|:-------------:|:--------------------------:|:---------------------------:|
| -1     | -1   | -             | 0.6605                     | -                           |
| 0.7331 | 500  | 0.3958        | -                          | -                           |
| 1.4663 | 1000 | 0.2682        | -                          | -                           |
| 2.1994 | 1500 | 0.2508        | -                          | -                           |
| 2.9326 | 2000 | 0.2166        | -                          | -                           |
| 3.6657 | 2500 | 0.1729        | -                          | -                           |
| -1     | -1   | -             | -                          | 0.8091                      |


### Training Time
- **Training**: 6.0 minutes

### Framework Versions
- Python: 3.10.9
- Sentence Transformers: 6.0.1
- Transformers: 5.16.1
- PyTorch: 2.14.0+cu126
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->