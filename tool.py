import sys

def view_bar(num, total):

  length = 30
  rate_num = round(num / total * length)
  r = '\r[%s%s]%d%%' % ("#"*(int)(rate_num), " "*(int)(length-rate_num), (num+1) / total * 100, )
  sys.stdout.write(r)
  sys.stdout.flush()

