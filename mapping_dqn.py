from environment import QubitAssignmentEnv
import matplotlib.pyplot as plt
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import os

num_iterations = 25000 # @param {type:"integer"}

initial_collect_steps = 250  # @param {type:"integer"}
collect_steps_per_iteration = 10# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}
log_interval = 50  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 100  # @param {type:"integer"}

num_qubits = 5
depth = 5

train_py_env = QubitAssignmentEnv(num_qubits, depth)
eval_py_env = QubitAssignmentEnv(num_qubits, depth)

#Wrap them in TF environments
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (32,32)

q_net = q_network.QNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    preprocessing_layers=tf.keras.layers.Flatten(),
    fc_layer_params=fc_layer_params,
    #dropout_layer_params=[None,0.25,0.2,None]
    )


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    emit_log_probability=False,
    debug_summaries = True,
    summarize_grads_and_vars = False,
    target_update_tau = 0.7,
    target_update_period=10,
    epsilon_greedy=1.0,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    print("Computing average return")
    total_return = 0.0
    for _ in range(num_episodes):
        print('Begining episode')
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            print(f'Chosen action: {action_step.action.numpy()}')
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

table_name = 'prioratized_table'
replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Prioritized(0.7),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbTrajectorySequenceObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2,
)

checkpoint_dir = os.path.join('checkpoint_bug_fixed')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    step_counter=agent.train_step_counter
)

# Restore from checkpoint
#train_checkpointer.initialize_or_restore()

#Run the random policy for a few steps to populate the replay buffer with data.
#@test {"skip": true}
py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

# Dataset generates trajectories with shape (Batchx2x...)
dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)


#Policy saver
policy_dir = os.path.join('policy_bug_fixed')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

# Reset the environment.
time_step = train_py_env.reset()

for i in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))
    # Save to checkpoint
    train_checkpointer.save(agent.train_step_counter.numpy())
    #If it doesn't work try this
    #train_checkpointer.save(optimizer.iterations.numpy())

    # Save policy
    tf_policy_saver.save(policy_dir)

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    #returns.append(avg_return)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
#plt.ylim(top=50)


# Load policy
#saved_policy = tf.saved_model.load(policy_dir)