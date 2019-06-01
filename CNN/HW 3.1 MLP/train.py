import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


def train(x_train, y_train, x_val, y_val, x_test, y_test, model, sess, log_file):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # batch size calculation
    num_instance = x_train.shape[0]
    num_batch = num_instance // FLAGS.batch_size

    # run required initializations
    sess.run(tf.global_variables_initializer())

    ####################################################################
    # TODO: Define Writers                                             #
    # Define 3 separate FileWriters for train, val and test sets       #
    # For more convenience use below log file names as logdir argument #
    # Do not forget to add graph to FileWriters                        #
    ####################################################################

    train_log_file = './logs/' + log_file + '/train/'
    val_log_file = './logs/' + log_file + '/val/'
    test_log_file = './logs/' + log_file + '/test/'

    train_writer = tf.summary.FileWriter(train_log_file, sess.graph)
    val_writer = tf.summary.FileWriter(val_log_file, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_file, sess.graph)

    ####################################################################
    #                         END OF YOUR CODE                         #
    ####################################################################

    # merge all summaries defined in a single summary
    merged_summary = tf.summary.merge_all()

    for epoch in range(FLAGS.num_epoch):
        for batch in range(num_batch):

            # select batch range
            batch_range = range(FLAGS.batch_size * batch, FLAGS.batch_size * (batch + 1))

            train_feed_dict = dict()
            #########################################################################
            # TODO: Feed Train Dictionary                                           #
            # Use update method of dictionary to feed model placeholders            #
            # Use batch range defined above  pointing to the range of current batch #
            #########################################################################
            train_feed_dict.update({model.placeholders['batch_images']: x_train[batch_range],
                                    model.placeholders['batch_labels']: y_train[batch_range]})
            #########################################################################
            #                         END OF YOUR CODE                              #
            #########################################################################

            # train
            sess.run(model.training, feed_dict=train_feed_dict)

        train_feed_dict = dict()
        ######################################################################################
        # TODO: Feed Train Dictionary and Run Session                                        #
        # Use update method of dictionary to feed model placeholders                         #
        # Feed all train images and labels                                                   #
        # Run session on merged_summary, loss and accuracy of model                          #
        # Use add_summary method of train_writer to add result of merged_summary evaluation  #
        # Note to set global step of summary based on epoch number                           #
        ######################################################################################

        train_feed_dict.update({model.placeholders['batch_images']: x_train,
                                model.placeholders['batch_labels']: y_train})

        train_summary, train_loss, train_acc = sess.run([merged_summary, model.loss, model.acc],
                                                        feed_dict=train_feed_dict)
        train_writer.add_summary(train_summary, global_step=epoch + 1)

        ######################################################################################
        #                                   END OF YOUR CODE                                 #
        ######################################################################################

        val_feed_dict = dict()
        ######################################################################################
        # TODO: Feed Validation Dictionary and Run Session                                   #
        # Use update method of dictionary to feed model placeholders                         #
        # Feed all validation images and labels                                              #
        # Run session just on merged_summary and loss of model                               #
        # Use add_summary method of val_writer to add result of merged_summary evaluation    #
        # Note to set global step of summary based on epoch number                           #
        ######################################################################################

        val_feed_dict.update({model.placeholders['batch_images']: x_val,
                              model.placeholders['batch_labels']: y_val})
        
        val_summary, val_loss = sess.run([merged_summary, model.loss], feed_dict=val_feed_dict)
        val_writer.add_summary(val_summary, global_step=epoch + 1)

        ######################################################################################
        #                                   END OF YOUR CODE                                 #
        ######################################################################################

        test_feed_dict = dict()
        ######################################################################################
        # TODO: Feed Test Dictionary and Run Session                                         #
        # Use update method of dictionary to feed model placeholders                         #
        # Feed all test images and labels                                                    #
        # Run session just on merged_summary                                                 #
        # Use add_summary method of test_writer to add result of merged_summary evaluation   #
        # Note to set global step of summary based on epoch number                           #
        ######################################################################################
        
        test_feed_dict.update({model.placeholders['batch_images']: x_test,
                              model.placeholders['batch_labels']: y_test})
        
        test_summary = sess.run(merged_summary, feed_dict=test_feed_dict)
        test_writer.add_summary(test_summary, global_step=epoch + 1)

        ######################################################################################
        #                                   END OF YOUR CODE                                 #
        ######################################################################################

        # print result of each epoch
        print('Epoch {}: train loss={:.3f}, train acc={:.3f}'.format(epoch + 1, train_loss, train_acc))
        print()

    test_feed_dict = dict()
    ###############################################################
    # TODO: Feed Test Dictionary and Run Session                  #
    # Use update method of dictionary to feed model placeholders  #
    # Feed all test images and labels                             #
    # Run session just on loss and accuracy of model              #
    ###############################################################
    
    test_feed_dict.update({model.placeholders['batch_images']: x_test,
                           model.placeholders['batch_labels']: y_test})

    test_loss, test_acc = sess.run([model.loss, model.acc], feed_dict=test_feed_dict)

    ###############################################################
    #                        END OF YOUR CODE                     #
    ###############################################################

    print('Test: average loss={:.3f}, average accuracy={:.3f}'.format(test_loss, test_acc))
    print('-------')
