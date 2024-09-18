import torch
import os
import argparse
from time import strftime, localtime
from torch.utils.tensorboard import SummaryWriter
from runners import runner
import xlwt

use_cuda = torch.cuda.is_available()


def main(arglist):
    current_time = strftime("%Y-%m-%d-%H-%M", localtime())
    #writer = SummaryWriter(log_dir='./logs/Charac_3s_vs_5z_r_False_' + current_time) #写日志数据
    #book=sheet_construct()#创建表格
    #book.save('E:\charac_data\map_3s_vs_5z_r_False.xls')
    writer = SummaryWriter(
        log_dir='./logs/' + arglist.algo_name + '/' + arglist.algo_name + '_' + arglist.scenario + arglist.r_F +'_'+ current_time)  # 写日志数据
    actors = 1  # 进程数
    if arglist.train == False:
        actors = 1
    env_runner = runner.Runner(arglist, arglist.scenario, actors)

    while arglist.train and env_runner.episode < 40001:
        env_runner.reset()
        # data_write(env_runner,sh0,sh1,sh2)
        replay_buffers = env_runner.run()  # 运行完一局战斗，并将数据存储在经验池（replaybuffer）中
        for replay_buffer in replay_buffers:
            env_runner.algo.episode_batch.add(replay_buffer)  # 把数据拷贝到迭代池中
        env_runner.algo.train()  # 训练
        for episode in env_runner.episodes:
            env_runner.algo.update_targets(episode)  # 更新目标网络

        for episode in env_runner.episodes:
            if episode % 400 == 0 and arglist.train:
                env_runner.algo.save_model(episode, env_runner.episode_global_step,
                                           './models/' + arglist.algo_name + '/' + arglist.algo_name + '_' + arglist.scenario  + arglist.r_F + '/agents_' + str(
                                               episode))  # 存储模型

        print(env_runner.win_counted_array)
        for idx, episode in enumerate(env_runner.episodes):
            print("episode: {}\nTotal reward: {}\nglobal step: {}".format(episode, env_runner.episode_reward[idx],
                                                                          env_runner.episode_global_step))
            if arglist.train:
                writer.add_scalar('Reward', env_runner.episode_reward[idx], episode)
                writer.add_scalar('Victory', env_runner.win_counted_array[idx], episode)

    test_num = 0
    win_Rate = 0
    test_limited = 2000
    while arglist.train == False and test_num < test_limited:
        env_runner.reset()
        replay_buffers = env_runner.run()  # 运行完一局战斗，并将数据存储在经验池（replaybuffer）中
        for replay_buffer in replay_buffers:
            env_runner.algo.episode_batch.add(replay_buffer)  # 把数据拷贝到迭代池中
        print(env_runner.win_counted_array)
        for idx, episode in enumerate(env_runner.episodes):
            win_Rate += env_runner.win_counted_array[idx]
            print("Win_count:", win_Rate)
            print("episode: {}\nTotal reward: {}".format(episode, env_runner.episode_reward[idx]))
            if not arglist.train:
                writer.add_scalar('test_Reward', env_runner.episode_reward[idx], episode)
                writer.add_scalar('test_Victory', env_runner.win_counted_array[idx], episode)
        test_num += 1
    win_Rate = win_Rate / test_limited
    print("\n\nwin_rate:", win_Rate)

    if arglist.train == False:
        env_runner.save()
    env_runner.close()


def sheet_construct():
    book = xlwt.Workbook(encoding='utf-8')
    sheet0 = book.add_sheet('brave')
    sheet1 = book.add_sheet('fear')
    sheet2 = book.add_sheet('coopration')
    return book


def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for DQN')

    parser.add_argument('--train', action='store_true')  # "3m"
    parser.add_argument('--load-episode-saved', type=int, default=8000)
    parser.add_argument('--scenario', type=str, default="2s3z")
    parser.add_argument('--algo_name', type=str, default="QMIX")
    parser.add_argument('--r_F', type=str, default="_0")

    return parser.parse_args()


if __name__ == "__main__":
    arglist = parse_args()

    try:
        os.mkdir('./models/' + arglist.algo_name + '/' + arglist.algo_name + '_' + arglist.scenario + arglist.r_F)
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")

    main(arglist)
