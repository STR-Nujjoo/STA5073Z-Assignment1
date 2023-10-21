# Tokenise by sentence
sona_sentences <- sona %>% 
  unnest_tokens(sentence, speech, token = 'sentences', to_lower = T) %>%
  dplyr::select(sentence, president_13, year) 

# Add a sentence id column
sona_sentences$Sid <- 1:nrow(sona_sentences)

# Find 200 most frequent word
word_bag_200 <- sona %>% 
  unnest_tokens(word, speech, token = 'regex', pattern = unnest_reg, to_lower = T) %>%
  filter(stringr::str_detect(word, '[a-z]')) %>%
  filter(!word %in% stop_words$word) %>%
  count(word)%>%
  top_n(200, wt = n) %>%
  select(-n)

# Bag of words model: Calculate number of times each of top 200 words was used in each sentence
# Drop any words from each sentence with none of the 200 top words.
bow_200 <- sona_sentences %>%
  unnest_tokens(word, sentence, token = 'regex', pattern = unnest_reg, to_lower = T) %>%
  filter(!word %in% stop_words$word, str_detect(word, '[a-z]')) %>%
  inner_join(word_bag_200) %>%
  group_by(Sid, word) %>%
  count() %>%
  ungroup() %>%
  left_join(sona_sentences %>% select(president_13, Sid)) %>% # bring back president corresponding to sentence id
  pivot_wider(names_from = word, values_from = n, values_fill = 0) %>%
  mutate(Pid = as.integer(factor(president_13))-1) %>% # Mandela = 0, Mbeki = 1, Ramaphosa = 2, Zuma = 3
  select(Pid, everything()) %>%
  select(-Sid)

set.seed(1)
bow_200 <- bow_200[sample(nrow(bow_200)),] # shuffle rows to make dataset more random

# table(bow_200$president_13)
# str(bow_200)

# Upsampled bag of words model
US_bow_200 <- as_tibble(upSample(bow_200, factor(bow_200$president_13)))
set.seed(1)
US_bow_200 <- US_bow_200[sample(nrow(US_bow_200)),] # shuffle rows to make dataset more random

# table(US_bow_200$president_13)
str(US_bow_200)
head(US_bow_200)

# Downsampled bag of words model
DS_bow_200 <- as_tibble(downSample(bow_200, factor(bow_200$president_13)))
set.seed(1)
DS_bow_200 <- DS_bow_200[sample(nrow(DS_bow_200)),] # shuffle rows to make dataset more random
# table(DS_bow_200$president_13)
# str(DS_bow_200)

#######################################################################################################################
# Applying tf-idf on the bag of words model
tfidf_200 <- sona_sentences %>%
  unnest_tokens(word, sentence, token = 'regex', pattern = unnest_reg, to_lower = T) %>%
  filter(!word %in% stop_words$word, str_detect(word, '[a-z]')) %>%
  inner_join(word_bag_200) %>%
  group_by(Sid, word) %>%
  count() %>%
  ungroup() %>%
  left_join(sona_sentences %>% select(president_13, Sid)) %>% # bring back president corresponding to sentence id
  bind_tf_idf(word, Sid, n) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
  mutate(Pid = as.integer(factor(president_13))-1) %>% # Mandela = 0, Mbeki = 1, Ramaphosa = 2, Zuma = 3
  select(Pid, everything()) %>%
  select(-Sid, -n, -tf, -idf)

set.seed(1)
tfidf_200 <- tfidf_200[sample(nrow(tfidf_200)),] # shuffle rows to make dataset more random


# table(tfidf_200$president_13)
# str(tfidf_200)

# Upsampled tf-idf model
US_tfidf_200 <- as_tibble(upSample(tfidf_200, factor(tfidf_200$president_13)))
set.seed(1)
US_tfidf_200 <- US_tfidf_200[sample(nrow(US_tfidf_200)),] # shuffle rows to make dataset more random
# table(US_tfidf_200$president_13)
# str(US_tfidf_200)

# Downsampled tf-idf model
DS_tfidf_200 <- as_tibble(downSample(tfidf_200, factor(tfidf_200$president_13)))
set.seed(1)
DS_tfidf_200 <- DS_tfidf_200[sample(nrow(DS_tfidf_200)),] # shuffle rows to make dataset more random
# table(DS_tfidf_200$president_13)
# str(DS_tfidf_200)

