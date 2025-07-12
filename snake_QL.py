import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle 
from matplotlib import style
import time
#from playsound import playsound

from collections import deque



# --- CONSTANTS AND CONFIG ---

style.use('ggplot')

# Environment settings
SIZE=20

# Q-Learning settings
EPISODES=20000
STEPS=1000
LEARNING_RATE=0.1
DISCOUNT=0.95

# Exploration settings
epsilon=0.9
eps_decay=0.9999

# Reward structure
MOVE_PENALTY=1
COLLIDE_PENALTY=300
FOOD_REWARD=25

# Visualization settings
show_every = 5000
start_q_table = None 



SNAKE_N=1
FOOD_N=2
BODY_N=3

# --- DICTIONARY FOR COLORS ---
d={
    1:(50, 200, 25),
    2:(255, 150, 0),
    3:(30, 144, 255)
}


class Snake:
    def __init__(self):
        self.x=np.random.randint(0,SIZE) 
        self.y=np.random.randint(0,SIZE)
        self.body=deque()
        self.growing=False


    def action(self, choice):
        if choice==0:
            self.move(x=0,y=1)
        elif choice==1:
            self.move(x=0,y=-1)
        elif choice==2:
            self.move(x=-1,y=0)
        elif choice==3:
            self.move(x=1,y=0)

    def move(self, x, y):

        self.body.appendleft((self.x,self.y))

        self.x+=x
        self.y+=y

        self.x=max(0,min(self.x,SIZE-1))
        self.y=max(0,min(self.y,SIZE-1))

        if not self.growing:
            if self.body:
                self.body.pop()

        else:
            self.growing=False




    def grow(self):
        self.growing=True


    def collide(self):
        return (self.x,self.y) in self.body


class Food:
    def __init__(self, snake=None):
        if snake.body is None:
            snake_body=set()
        else:
            snake_body=snake.body

        self.respawn(snake_body)


    def respawn(self, snake_body):
        while True:
            self.x=np.random.randint(0,SIZE) 
            self.y=np.random.randint(0,SIZE)
            
            if (self.x,self.y) not in snake_body:
                break




def get_q_table(path):

    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    else:
        q_table = {}
        for x1 in [-1,0,1]:
            for y1 in [-1,0,1]: 
                for d_l in [True, False]:
                    for d_r in [True, False]:
                        for d_u in [True, False]:
                            for d_d in [True, False]:
                                state = (x1, y1, d_l, d_r, d_u, d_d)
                                q_table[state] = [np.random.uniform(-5, 0) for _ in range(4)]

    return q_table



def get_state(snake, food):

    dx=np.sign(snake.x-food.x)
    dy=np.sign(snake.y-food.y)

    body_set=set(snake.body)

    pl=(snake.x-1,snake.y)
    pr=(snake.x+1,snake.y)
    pd=(snake.x,snake.y+1)
    pu=(snake.x,snake.y-1)

    danger_l=pl[0]<0 or pl in body_set
    danger_r=pr[0]>=SIZE or pr in body_set
    danger_u=pu[1]<0 or pu in body_set
    danger_d=pd[1]>=SIZE or pd in body_set

    return (dx,dy,danger_l,danger_r, danger_u, danger_d)

def draw_grid(image, size, color=(200, 200, 200)):
    step = image.shape[0] // size
    for i in range(0, image.shape[0], step):
        cv2.line(image, (i, 0), (i, image.shape[1]), color, 1)
        cv2.line(image, (0, i), (image.shape[1], i), color, 1)
    return image

def draw_environment(food, snake, episode, episode_reward, state_flag):
    env=np.zeros((SIZE,SIZE,3), dtype=np.uint8)
    env[food.y][food.x]=d[FOOD_N]
    env[snake.y][snake.x]=d[SNAKE_N]
    
    for (ax,ay) in snake.body:
        env[ay][ax]=d[BODY_N]


    img=Image.fromarray(env, "RGB")
    img=img.resize((700, 700), Image.BICUBIC)
    
    img = np.array(img)
    img = draw_grid(img, SIZE)
    cv2.putText(img, f"Ep: {episode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Len: {len(snake.body)+1}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Reward: {episode_reward}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Mode: {state_flag}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Snake Q-Learn",img)

    if cv2.waitKey(30) & 0xFF==ord('q'):
        return False

    return True



if __name__=="__main__":

    q_table=get_q_table(start_q_table)
    episode_rewards=[]


    for episode in range(EPISODES):
        snake=Snake()
        food=Food(snake)

        show=(episode%show_every==0)

        if show:
            print(f"## ON EPISODE {episode}, EPSILON: {epsilon}")
    

        episode_reward=0
        done=False

        

        for i in range(STEPS):
            obs=get_state(snake,food)

            if np.random.random()>epsilon:
                action=np.argmax(q_table[obs])
            else:
                action=np.random.randint(0,4)

            distance_before=abs(snake.x-food.x)+ abs(snake.y-food.y)

            snake.action(action)

            distance_after=abs(snake.x-food.x)+ abs(snake.y-food.y)

                       
            state_flag='SEARCHING...'
            if snake.collide():
                state_flag='COLLISION'
                reward=-COLLIDE_PENALTY
                done=True
                
                
            elif snake.x==food.x and snake.y==food.y:
                state_flag='EATING'
                reward=FOOD_REWARD
                snake.grow()
                food.respawn(set(snake.body))
                
                
            else:
                if distance_after>distance_before:    
                    reward=-MOVE_PENALTY*2
                else:
                    reward=-MOVE_PENALTY
                

            new_obs=get_state(snake,food)
            max_future_q=np.max(q_table[new_obs])
            current_q=q_table[obs][action]

           
            new_q=(1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[obs][action]=new_q

            episode_reward+=reward
                

            if show:
                if not draw_environment(food, snake, episode, episode_reward, state_flag):
                    break

        episode_rewards.append(episode_reward)
        epsilon*=eps_decay

    moving_avg = np.convolve(episode_rewards, np.ones((show_every,))/show_every, mode='valid')
    plt.figure(figsize=(10,6))
    plt.plot(range(len(moving_avg)), moving_avg)
    plt.title("Q-Learning Training Progress")
    plt.ylabel(f"Reward ({show_every} Episode Moving Average)")
    plt.xlabel("Episode #")
    plt.grid(True)
    plt.show()

    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)


