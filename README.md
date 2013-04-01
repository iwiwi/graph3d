Graph3d
======================


使い方
----------------
### ビルド ###
    $ ./waf configure
    $ ./waf
 
freeglut が必要です．Linux で動作確認をしていますが，Mac でも動作の報告があります．

### 起動 ###
    $ bin/graph3d --graph=GRAPH_FILE

入力のグラフは，各行に 1 本の辺を表す 2 整数を空白で区切って書く形式で用意して下さい．

起動後は，左ドラッグで回転，右ドラッグでズーム，`A` キーと `Z` キーで透明度変更です．

 
例
----------------

![画像1](http://www-imai.is.s.u-tokyo.ac.jp/~takiba/img/graph_cagrqc_w200.png)
　![画像2](http://www-imai.is.s.u-tokyo.ac.jp/~takiba/img/graph_reactome_w200.png)
　![画像3](http://www-imai.is.s.u-tokyo.ac.jp/~takiba/img/graph_arxiv_w200.png)


謝辞
--------

* [wata-orz](https://github.com/wata-orz)
