import random
from game import move_action2move_id, Game, Board
from mcts import MCTSPlayer
from config import CONFIG

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('Unsupported framework')


# 测试Board中的start_play
class Human1:
    def get_action(self, board):
        move = move_action2move_id[input('Please input')]
        # move = random.choice(board.availables)
        return move

    def set_player_ind(self, p):
        self.player = p


def main():
    if CONFIG['use_frame'] == 'paddle':
        policy_value_net = PolicyValueNet(model_file='current_policy.model')
    elif CONFIG['use_frame'] == 'pytorch':
        policy_value_net = PolicyValueNet(model_file='current_policy.pkl')
    else:
        print('Unsupported framework')

    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                    c_puct=5,
                                    n_playout=100,
                                    is_selfplay=0)

    human = Human1()

    game = Game(board=Board())
    game.start_play(mcts_player, human, start_player=1, is_shown=1)


if __name__ == '__main__':
    main()
