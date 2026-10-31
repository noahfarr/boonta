# Step budgets: every epoch runs a whole number of updates of 4096 x 16 = 65,536 steps, and every run logs 61 points.
PPO_BUDGET="num_steps=19988480 num_epochs=61"   # 305 updates = 61 epochs x 5
PQN_BUDGET="num_steps=79953920 num_epochs=61"   # 1220 updates = 61 epochs x 20
GAMES="breakout asterix space_invaders freeway seaquest"
