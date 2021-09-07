saver = tf.train.Saver()
saver.restore(sess, 'model.ckpt')

val_new = open("/home/mukesh/dl_rnn/WeatherGov/dev/dev.combined", "r")
val_new = val_new.read()
val_new = val_new.split('\n')

fd = next_feedval(1)
l = sess.run([alphas], fd)


words  = val_new[0].split(' ')

pp1=np.asarray(l)

pp1= pp1.reshape(107) #length

with open("visualization2.html", "w") as html_file:
    for word, alpha in zip(words, pp1 / pp1.max()):
        html_file.write('<font style="background: rgba(250, 250, 50, %f)">%s</font>\n' % (alpha, word))