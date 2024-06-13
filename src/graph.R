rm(list=ls())
library(dplyr)
library(xlsx)
library(readxl)
library(matrixcalc)
library(reshape2)
library(tidytext)
library(stringr)
library(syuzhet)
library(tidyr)
library(readr)
library(visNetwork)
library(tm)
library(widyr)
library(textstem)
library(igraph)
library(ape)
library(RColorBrewer)
library(plotly)
library(readxl)

setwd('C:/Users/melanie.bernal/OneDrive - Ipsos/Ipsos/Estudios/24-008489-01 PreTest Red/BASES')
proyecto = '24-008489-01 PreTest Red'

col_names <- 'Frase Corregida'

for(col_name in col_names){
  
  #------------------------------------------------------------------------------
  diam <- read_excel(paste0('diam-',col_name,'.xlsx')) %>% as.data.frame()
  colnames(diam)[1] <- 'word'
  #------------------------------------------------------------------------------
  grafo <- read_excel(paste0('grafo-',col_name,'.xlsx')) %>% as.data.frame()
  
  #eliminar <- read_excel('Modificaciones.xlsx',sheet='eliminar')
  #cambios <- read_excel('Modificaciones.xlsx',sheet='cambios')
  #eliminar <- cambios <- NULL
  
  #***********************#
  #****** FUNCIONES ******#
  #***********************#
  
  grafo_ <- function(graph, diam, vertex_color='#00BFFF', vertex_size=c(25,75)){
    
    descompose_net.data = decompose(graph)
    cual = which(unlist(lapply(descompose_net.data , vcount)) >= 5)
    descompose_net.data = descompose_net.data[cual]
    
    nodes = edges = NULL
    
    for(descompose_net.data_i in descompose_net.data){
      
      graph.net <- toVisNetworkData(descompose_net.data_i)
      nodes_i = graph.net$nodes; edges_i = graph.net$edges
      
      nodes = rbind(nodes,nodes_i)
      edges = rbind(edges,edges_i)
      
    }
    
    nodes = merge(nodes,diam,by.x='id',by.y='lema')
    colnames(nodes)[3] = 'size'
    
    importance = nodes$size/sum(nodes$size)
    Imp = data.frame(palabra = nodes$id, importance=importance, menciones = nodes$size)
    
    nodes$size <- scales::rescale(nodes$size,c(25,75))
    nodes$color <- vertex_color
    
    edges$width <- scales::rescale(edges$weight,to = c(1,10))
    
    return(list(nodes=nodes, edges = edges, Imp = Imp))
    
  }
  
  # Cambios ----------------------------------------------------
  cambios_grafo = function(grafo,word,change){
    
    for(i in 1:length(word)){
      
      grafo$from[which(grafo$from == word[i])] = change[i]
      grafo$to[which(grafo$to == word[i])] = change[i]
      
    }
    
    for(i in 1:nrow(grafo)){
      
      grafo_i = grafo[i,-3] %>% sort
      grafo[i,-3] =  grafo_i
      
    }
    
    grafo = grafo %>% group_by(from, to) %>% 
      summarise(sim = max(sim))
    
    grafo = grafo %>% filter(from!=to) %>% as.data.frame()
    
    return(grafo)
    
  }
  
  cambios_diam = function(diam,word,change){
    
    
    for(i in 1:length(word)){
      
      cual = which(diam$word == word[i])
      diam$word[cual] = change[i]
      
    }
    
    diam = diam %>% group_by(word) %>% summarise(size = sum(size)) %>% as.data.frame()
    
    return(diam)
    
  }
  
  # ------------------------------------------------------------
  
  #grafo <- cambios_grafo(grafo,NULL,NULL)
  #grafo <- cambios_grafo(grafo, cambios$word, cambios$cambio)
  grafo %>% filter(from == to)
  grafo[,1:2] %>% duplicated() %>% sum
  
  nrow(diam)
  #diam <- cambios_diam(diam,cambios$word, cambios$cambio)
  nrow(diam)
  
  #------------------------------------------------------------------------------
  
  menciones <- 3
  diam <- diam %>% filter(size >= menciones)
  
  grafo <- grafo %>% 
    filter(from %in% diam$word) %>%
    filter(to %in% diam$word)
    #filter(! from %in% eliminar$eliminar) %>%
    #filter(! to %in% eliminar$eliminar)
  
  colnames(grafo) <- c('item1','item2','correlation')
  colnames(diam) <- c('lema','size')
  
  grafo$item1 <- str_to_title(grafo$item1)
  grafo$item2 <- str_to_title(grafo$item2)
  diam$lema <-  str_to_title(diam$lema)
  
  grafo <- grafo %>% filter(correlation > 0.5)
  
  #------------------------------------------------------------------------------
  
  grafo$weight <- grafo$correlation/sum(grafo$correlation)
  grafo$width <- scales::rescale(grafo$weight,to = c(1,10))
  graph <- graph.data.frame(grafo, directed = FALSE)
  
  graph_info <- grafo_(graph, diam = diam) 
  nodes = graph_info$nodes
  edges = graph_info$edges
  
  redes = visNetwork(nodes, edges,submain = proyecto,
                     main = 'Text Analytics') %>%
    visEdges(physics = TRUE) %>%
    visNodes(font = list(size = 70)) %>%
    visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
    visPhysics(solver = "forceAtlas2Based", 
               forceAtlas2Based = list(gravitationalConstant = -100)) 
  
  redes
  setwd('../SALIDA')
  visSave(redes, paste0(proyecto,'.html'))
  
  # AN?LISIS DESCRIPTIVO
  #-------------------------------------------------------------------------------------------
  net.data = graph.data.frame(edges, directed = FALSE)
  transitivity(net.data)
  
  #*-------------------------------------------------------------
  #* Partitioning
  #*-------------------------------------------------------------
  
  kc <- cluster_fast_greedy(net.data)
  length(kc)
  sizes(kc)
  group = membership(kc)
  cluster = data.frame(id = names(group), group = as.numeric(group))
  plot(kc,net.data)
  
  #colores = data.frame(color = brewer.pal(length(kc), 'Oranges'),
  #                     group = 1:length(kc))
  #colores$color[1] = 'yellow'
  
  Imp = graph_info$Imp
  nodes = merge(nodes,cluster)
  nodes = merge(nodes,Imp,by.x = 'id',by.y = 'palabra')
  #nodes = merge(nodes %>% dplyr::select(-color),colores,by='group')
  
  colnames(nodes)
  write.csv2(nodes %>% 
               dplyr::select(-color,-size),
             paste0(proyecto,'-Importancias-',col_name,'.csv'),row.names = F)
  
  (partition <- visNetwork(nodes %>% 
                             dplyr::select(-importance,
                                           -menciones,-color) %>%
                             arrange(id), edges, 
                           main = 'Partición Grafo Total',submain = proyecto) %>%
      visNodes(font = list(size = 70)) %>%
      visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%   
      visEdges(physics = TRUE) %>%
      visPhysics(solver = "forceAtlas2Based",
                 forceAtlas2Based = list(gravitationalConstant = -150,
                                         stabilization = T)))
  
  visSave(partition, paste0(proyecto,'_clasificado_',col_name,'.html'))
  setwd('../BASES')
}

