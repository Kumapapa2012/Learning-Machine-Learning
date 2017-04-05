#
# -*- coding: utf-8 -*- 
import numpy as np


class Game_Reversi:
    turn = 1      #agent=1,environment=-1
    g_board=[]
    
    n_rows = 6
    n_cols = 6
    
    # 報酬
    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0
    
    # 表示
    render=True
    
    #__init__(n_rows,n_cols)
    #    n_rows: ボードサイズ縦
    #    n_cols: ボードサイズ横
    #指定されたボードサイズでクラスを作成。
    def __init__(self,n_rows,n_cols):
        #ボードリセット
        self.n_rows = n_rows
        self.n_cols = n_cols
        
        self.g_board=np.zeros([self.n_rows,self.n_cols],dtype=np.int16)
        # オセロ、中心に最初の4コマを置く
        self.g_board[self.n_rows//2-1,self.n_cols//2-1]=1
        self.g_board[self.n_rows//2-1,self.n_cols//2]=-1
        self.g_board[self.n_rows//2,self.n_cols//2-1]=-1
        self.g_board[self.n_rows//2,self.n_cols//2]=1
    
    #isValidMove(self,row,column,c)
    #    row: コマの位置縦
    #    col: コマの位置横
    #    c:対象プレイヤー(1=Agentもしくは人間、-1=Environment)
    #指定された row,col が、c にとって正しい手か判定。コマを置けるかの判定用。
    #一筋でも相手のコマを返せるならば true とする
    #Return: 正しい手=True、無効な手=False
    def isValidMove(self,row,column,c):
        #
        # Phase 0: すでにコマがあるか
        #
        if (self.g_board[row,column] != 0):
            return False
        
        result=False
        pos=np.array([row,column])    
        # 指定場所から全方向にチェック。
        # 返せる相手のコマが一つでもあれば終了
        for direction in ([-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]):
            if not (0 <= (pos+direction)[0] < self.n_rows and 0 <= (pos+direction)[1] < self.n_cols ) :
                # 範囲外処理スキップ
                continue
            
            #
            # Phase 1: 隣接するコマの色が指定色の反対か？
            #
            cpos = pos + direction
            if (self.g_board[tuple(cpos)] == -c):
                #
                # Phase 2: その向こうに自コマがあるか
                #
                while (0 <= cpos [0] < self.n_rows and 0 <= cpos [1] < self.n_cols):
                    if (self.g_board[tuple(cpos)] == 0):
                        # 判定がつく前に空のマスなので終了
                        break
                    elif (self.g_board[tuple(cpos)] == -c):
                        # 自コマがこの先あれば、取れる可能性のあるコマ。
                        # 自コマの探索を続ける。
                        cpos = cpos+direction
                        continue
                    elif (self.g_board[tuple(cpos)] == c):
                        # 少なくともコマを一つ返せるため、この時点で探索終了。
                        result=True
                        break
                    else:
                        print("catastorophic failure!!! @ isValidMove")
                        exit()
        return result
    
    #putStone(row,column,c,sim)
    #    row: コマの位置縦
    #    col: コマの位置横
    #    c: 対象プレイヤー(1=Agentもしくは人間、-1=Environment)
    #    sim: 結果をゲームに反映するか(False=ゲームに反映する、True=ゲームに反映しない)。
    #前提条件： すでに、isValidMove で正しい位置であることが確認済。
    #コマを置いて、相手のコマを返す。
    #第 4 引数 sim(ulation) == True とするとコピーのボードで処理し、実際のボードはタッチしない。
    #Return: 返した相手のコマの数
    def putStone(self,row,column,c,sim):
        if sim :
            board=np.copy(self.g_board)
        else:
            board=self.g_board
        
        numFlip = 0 # 返した数
        pos=np.array([row,column])
        
        # コマを置く
        board[tuple(pos)]=c
        
        # 指定場所から全方向にチェック。
        # [-1,-1],[-1,0],[-1,1]
        # [ 0,-1],      ,[ 0,1]
        # [ 1,-1],[ 1,0],[ 1,1]
        # 返せるコマはすべて返す。
        for direction in ([-1,-1],[-1,0],[-1,1],[ 0,-1],[ 0,1],[ 1,-1],[ 1,0],[ 1,1]):
            f_canFlip=False
            cpos = pos + direction
            while 0 <= cpos [0] < self.n_rows and 0 <= cpos [1] < self.n_cols:
                if (board[tuple(cpos)] == -c):  
                    #相手のコマがある。
                    #同方向の次のコマで探索を継続。
                    cpos=cpos+direction
                    continue
                elif (board[tuple(cpos)] == c): 
                    #自コマで相手コマを挟める。
                    #この場所から、相手のコマを返す。
                    f_canFlip=True
                    break
                elif (board[tuple(cpos)] == 0): 
                    # この方向では相手コマを挟めない。
                    break
                else:
                    print("catastorophic failure!!! @ putStone")
                    exit()
            
            # 現在の cpos の位置から、指定した位置まで後進してコマを返す。
            if f_canFlip :
                cpos=cpos-direction #返すコマ一つ目
                while np.array_equal(cpos,pos) == False:
                    board[tuple(cpos)] = c
                    numFlip = numFlip + 1
                    cpos=cpos-direction
        return numFlip
    
    
    #resetBoard()
    #ボードをゲーム初期状態に戻す。
    def resetBoard(self):
        self.g_board=self.g_board*0
        #ボードリセット
        self.g_board[self.n_rows//2-1,self.n_cols//2-1]=1
        self.g_board[self.n_rows//2-1,self.n_cols//2]=-1
        self.g_board[self.n_rows//2,self.n_cols//2-1]=-1
        self.g_board[self.n_rows//2,self.n_cols//2]=1
    
    #getPositionAvail(c)
    #    c: 対象プレイヤー(1=Agentもしくは人間、-1=Environment)
    #現時点の盤面で、対象プレイヤーがコマを置ける=相手のコマを返すことができる場所を探す。
    #Return: コマを置ける場所のリスト
    def getPositionAvail(self,c):
        temp=np.vstack(np.where(self.g_board==0))
        nullTiles=np.hstack((temp[0].reshape(-1,1),temp[1].reshape(-1,1)))    
        
        # コマが無いマスについて、 IsValidMove()
        can_put=[]
        for p_pos in nullTiles:
            if self.isValidMove(p_pos[0],p_pos[1],c):
                can_put.append(p_pos)
        return can_put

    #getPosition(c)
    #    c: 対象プレイヤー(1=Agentもしくは人間、-1=Environment)
    #現時点の盤面でコマを置ける場所のうち、以下のルールで置くべき場所を返す。
    #1. 四隅にコマが置ける場合は、90% の確率でそこを取る
    #2. 他の場合は、80% の確率で最も返せるコマの数が多いものを取得する。
    #3. いずれでも決まらない場合ランダム(結局　1,2　のものとなる可能性もある。)
    #Return: コマを置くべき場所
    def getPosition(self,c):
        can_put=[]
        can_put=self.getPositionAvail(c)
        
        maxPos=[]
        cornerPos=[]
        returnPos=[]
        numStone=0
        for p_pos in can_put:
            if (p_pos[0]==0 or p_pos[0]==self.n_rows-1) and (p_pos[1]==0 or p_pos[1]==self.n_cols-1):
                cornerPos.append(p_pos)
            cur_numstone=self.putStone(p_pos[0],p_pos[1],c,True)
            if numStone < cur_numstone:
                numStone = cur_numstone
                maxPos=p_pos
        
        #　can_put　が空ならこの時点でreturn。
        # 結果的に PASS　になる。
        if can_put==[] :
            return []
        
        # ランダムとするか決める
        t_rnd=np.random.random()
        
        # 1. 角がある場合は、90% の確率でそこを取る
        if cornerPos != []:
            if t_rnd < 0.9:
                returnPos= cornerPos[np.random.randint(0,len(cornerPos))]
        
        #　2. 次に、80% の確率で最も数が多いものを取得する。
        if returnPos==[]:
            if maxPos != []:
                if t_rnd < 0.8:
                    returnPos= maxPos
        
        #　3. この時点で決まらない場合ランダム(結局　1,2　のものとなる可能性あり。)
        if returnPos==[]:
            returnPos= can_put[np.random.randint(0,len(can_put))]
        
        return returnPos
        
    #step(player_pos)
    #    player_pos: Agentもしくは人間がコマを打つ位置
    #1. 受け取った位置に相手のコマを置き、ボードに反映する。
    #2. 自身のコマを置きボードに反映する。
    #3. 報酬の決定、終了判定を行う
    #Return: ボード状態、報酬、終了かどうか
    def step(self,player_pos):
        #　パスしたか?    
        if player_pos ==(-1,-1) :
            # 今は特に処理なし。。
            if self.render : print("PASS!")
        else:
            if(self.isValidMove(player_pos[0],player_pos[1],self.turn)):
                # 更新
                self.putStone(player_pos[0],player_pos[1],self.turn,False)
            else:
                # トレーニングとValidation中は発生し得ない
                # reward0 でそのまま返す。
                print("Bad Move...%i, %i, %i"%(player_pos[0],player_pos[1],self.turn))
                return self.g_board,0,False
        
        # 盤表示
        if self.render : print("Your turn ends...")
        if self.render : print(self.g_board)
        
        # ****環境のターン
        stonePos = self.getPosition(-self.turn)
        if stonePos == []:
            # 今は特に処理なし。。
            if self.render : print("PASS!")
        else :
            # コマを置いて返す
            self.putStone(stonePos[0],stonePos[1],-self.turn,False)
        if self.render : print("My turn ends...")
        if self.render : print(self.g_board)
        
        #
        # 終了判定、報酬の計算
        #
        reward=0.
        done=False
        
        # ****双方のターン終了のため勝敗判定
        # 1.　エージェントまたは環境の双方における場所があるか？
        # なければ終了
        stonePos_agent = self.getPosition(self.turn)
        stonePos_environment = self.getPosition(-self.turn)
        if stonePos_agent==[] and stonePos_environment==[]:
            done=True
            if self.render : print("****************Finish****************")
        
        if done :
            #　2.　終了ならば報酬計算
            num_agent=len(np.where(self.g_board==self.turn)[1])       #　[1]指定は、2　x(コマ数)の配列になるため
            num_envionment=len(np.where(self.g_board==-self.turn)[1]) #　[1]指定は、2　x(コマ数)の配列になるため
            if self.render : print("you:%i/environment:%i" % (num_agent,num_envionment))
            # 判定
            if num_agent > num_envionment :
                reward=1.0
                if self.render : print("you win!")
            elif num_agent < num_envionment :
                reward=-1.0
                if self.render : print("you lose!")
            else :
                reward=-0.5
                if self.render : print("Draw!")
        #終了    
        # observation, reward, done を返す
        return self.g_board,reward,done

