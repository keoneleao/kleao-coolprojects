# Cipher-SR
### Behavioral Cloning for Support & Resistance Trading
Cipher-SR is an imitation learning system that learns a trader's decision-making process from demonstrations. Users interact with an annotation GUI to place support/resistance levels, Fibonacci retracements, and buy/sell actions, which are converted into state-action demonstrations used to train a neural network policy.

(GUI gif)

### Why I Built This

I decided to investigate a section of daily life that does not require expensive robot hardware but had a meaningful and impactful effect. Trading is difficult to automate as it requires human discretion and knowledge. Automating trading with an AI capable of making trustworthy decisions would provide a passive stream of income for people around the world. 

### System Architecture

(diagram)

### Features

✅ Interactive demonstration GUI

✅ Custom candlestick renderer

✅ Support/Resistance annotations

✅ Fibonacci retracement annotations

✅ Sequential behavioral cloning

✅ Feature normalization

✅ Class balancing

✅ Replay visualization

### Machine Learning Pipeline

(diagram)

### State Vector (or State Vector?)

|State|Purpose|
|---|---|
|log_close|Normalize prices across stocks|
|return_1|Short-term momentum|
|return_3|Medium-term momentum|
|SR_rel|Relative distance to SR line|
|SR_exists|Indicates active SR level|

### Neural Network

(diagram)

### Data Collection GUI

(image w/ annotations)

### Replay GUI

(sidebyside video (human vs AI))

### Technical Challenges & Engineering Decisions

Why SR Lines?
Support and resistance is the basis for many trading strategies. It is a great tool to transfer a traders ideology to the AI trading agent. 

Why use AI?
It is possible to make a trading agent that is hardcoded (ex: buy when price touches support); however, it is important that people trust their trading agent. Cipher acts as the user by replicating their trading strategy. Who better to trust than yourself?

Why CrossEntropyLoss?
MSELoss is usually utilize when numbers have continous physical meaning (ex: distance). CrossEntropyLoss is better for discrete actions (ex: buy, hold, sell), which becomes more robust when paired with Dropout. 

Class imbalace: Most actions are HOLD
Buying/Selling is done instantaneously, while Holding takes up most time when trading. To combat the HOLD action dominance, I downsampled HOLD actions and applied a weighted CrossEntropyLoss.

Generalization: Raw stock prices aren't generalizable
Stocks have different raw prices and move different amounts. Cipher applies to a broader range of stocks once trained due to the normalized prices and relative distance to SR levels. 

Temporal Context: A single price lacks historical context
'The stock is priced at 100$' gives no imformation. This is why I put four sequentail states into one policy input. The sequential input gives the model context. 'The stock quickly rose to 100$ over the past four states.' 

Irrational Trading: Model is selling lower than it brought
The model would buy multiple times before selling. Additionally, the model would sell lower than it brought. These actions are utilized by traders in different circumstances; however, these actions were not trained by the user, and they were caused by the AI's lack of context. To keep the AI trader simple and effective, I added constraints to improve performance. (ex: Do not buy if holding a stock and do not sell lower than buy price)

### Results



### Technologies

(badges)

### Cipher Folder Structure

(diagram/text)

### Future Improvements

(list)

### What I Learned

(text)

