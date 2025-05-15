from board import Board
from solver import solve

def play_game():
    board = Board()
    print(board)
    while True:
        
        
        
        player_move = int(input("Nhập cột (0-6): "))
        if not board.can_play(player_move):
            print("Nước đi không hợp lệ, thử lại!")
            continue
        board.play(player_move)
        print(board)
        if board.winning_board_state():
            print("Bạn thắng!")
            break

        # AI chơi
        ai_move = solve(board)
        print(f"AI chọn cột: {ai_move}")
        board.play(ai_move)
        print(board)
        if board.winning_board_state():
            print("AI thắng!")
            break


        # Người chơi nhập nước đi
        

        
        
        # ai2_move = solve(board)
        # print(f"AI2 chọn cột: {ai2_move}")
        # board.play(ai2_move)
        # print(board)
        # if board.winning_board_state():
        #     print("AI2 thắng!")
        #     break


        
        
        
        
        if board.moves == board.w * board.h:
            print("Hòa!")
            break

if __name__ == "__main__":
    play_game()
